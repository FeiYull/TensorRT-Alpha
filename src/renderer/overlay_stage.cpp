// =============================================================================
//  overlay_stage (implementation)
// =============================================================================
#include "overlay_stage.hpp"

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/depth/types.hpp"
#include "trt_alpha/kernels/ops/overlay.hpp"
#include "trt_alpha/seg/types.hpp"
#include "trt_alpha/sem/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace trt_alpha::renderer {
namespace {

//! ultralytics Colors(i, bgr=True) -- 20 entries, RGB. Kept identical to the
//! host renderer's palette so the CUDA composite picks the same colours.
constexpr std::uint8_t kUltraPalette[20][3] = {
    {  4,  42, 255}, { 11, 219, 235}, {243, 243, 243}, {  0, 223, 183},
    { 17,  31, 104}, {255, 111, 221}, {255,  68,  79}, {204, 237,   0},
    {  0, 243,  68}, {189,   0, 255}, {  0, 180, 255}, {221,   0, 186},
    {  0, 255, 255}, { 38, 192,   0}, {  1, 255, 179}, {125,  36, 255},
    {123,   0, 104}, {255,  27, 108}, {252, 109,  47}, {162, 255,  11},
};

//! The box the mask is confined to, widened by a couple of pixels so the
//! bilinear transition at the crop border is still evaluated exactly like the
//! host code did (which sampled the whole frame).
constexpr int kBoxPad = 3;

int padClamp(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

//! numpy.percentile(v, q) with method="linear", via nth_element (O(n) instead of
//! a full sort). `v` is permuted; the caller never relies on its order.
float numpyPercentile(std::vector<float>& v, double q)
{
    if (v.empty())
    {
        return 0.0f;
    }
    const double pos = (q / 100.0) * static_cast<double>(v.size() - 1);
    const double lo  = std::floor(pos);
    const std::size_t i0 = static_cast<std::size_t>(lo);
    const std::size_t i1 = std::min<std::size_t>(i0 + 1, v.size() - 1);

    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i0), v.end());
    const float x0 = v[i0];
    float x1 = x0;
    if (i1 != i0)
    {
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i1), v.end());
        x1 = v[i1];
    }
    const double frac = pos - lo;
    return static_cast<float>(static_cast<double>(x0) * (1.0 - frac) +
                              static_cast<double>(x1) * frac);
}

//! The 2% / 98% disparity range that colorizeDepth() normalises against.
//! This is a reduction over the depth map, not a paint, so it stays on the host.
void disparityRange(const depth::DepthMap& dm, float& vmin, float& vmax)
{
    std::vector<float> pool;
    pool.reserve(dm.data.size());
    for (float d : dm.data)
    {
        if (d > 0.0f)
        {
            pool.push_back(1.0f / d);
        }
    }
    vmin = 0.0f;
    vmax = 1.0f;
    if (!pool.empty())
    {
        vmin = numpyPercentile(pool, 2.0);
        vmax = numpyPercentile(pool, 98.0);
    }
    if (vmax <= vmin)
    {
        vmax = vmin + 1e-6f;
    }
}

}  // namespace

struct OverlayStage::Impl
{
    cudaStream_t     stream = nullptr;
    core::DeviceBuffer frame;
    core::DeviceBuffer layers;
    core::DeviceBuffer masks;
    core::DeviceBuffer sem;
    core::DeviceBuffer depth;
    core::PinnedBuffer hostOut;

    std::vector<kernels::ops::SegLayer> layerHost;
    const char* lastError = nullptr;

    bool ensureStream()
    {
        if (stream != nullptr)
        {
            return true;
        }
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        {
            lastError = "cudaStreamCreate failed";
            return false;
        }
        return true;
    }

    void note(cudaError_t e)
    {
        if (e != cudaSuccess)
        {
            lastError = cudaGetErrorString(e);
            (void)cudaGetLastError();
        }
    }
};

OverlayStage::OverlayStage() : m_impl(std::make_unique<Impl>()) {}

OverlayStage::~OverlayStage()
{
    if (m_impl->stream != nullptr)
    {
        cudaStreamSynchronize(m_impl->stream);
        cudaStreamDestroy(m_impl->stream);
        m_impl->stream = nullptr;
    }
}

const char* OverlayStage::lastError() const noexcept
{
    return m_impl->lastError;
}

// -----------------------------------------------------------------------------
//  Device composite
// -----------------------------------------------------------------------------
const std::uint8_t* OverlayStage::composeDevice(const core::BatchResult& result,
                                                const std::uint8_t* hostBgr,
                                                int width,
                                                int height,
                                                std::size_t stride,
                                                float maskAlpha,
                                                int index)
{
    if (hostBgr == nullptr || width <= 0 || height <= 0 || index < 0)
    {
        return nullptr;
    }

    const std::size_t idx = static_cast<std::size_t>(index);

    // ---- 1. collect the mask layers (host description + device copies) ------
    m_impl->layerHost.clear();

    const bool hasSeg = result.segmentations.size() > idx &&
                        !result.segmentations[idx].empty();
    const bool hasSem = result.semantics.size() > idx && !result.semantics[idx].empty();
    const bool hasDepth = result.depths.size() > idx && !result.depths[idx].empty();
    if (!hasSeg && !hasSem && !hasDepth)
    {
        return nullptr;
    }

    // Same order the host renderer painted in: masks -> semantic -> depth.
    static const std::vector<seg::Segmentation> kNoSegs;
    const auto& segs = hasSeg ? result.segmentations[idx] : kNoSegs;

    std::size_t maskBytes = 0;
    for (const auto& s : segs)
    {
        if (s.mask.data == nullptr || s.mask.width <= 0 || s.mask.height <= 0 ||
            s.mask.channels != 1)
        {
            continue;
        }
        maskBytes += static_cast<std::size_t>(s.mask.height) *
                     static_cast<std::size_t>(s.mask.stride);
    }
    if (!m_impl->ensureStream())
    {
        return nullptr;
    }
    cudaStream_t st = m_impl->stream;

    // ---- 2. stage the frame + the masks on the device ----------------------
    const std::size_t frameBytes = stride * static_cast<std::size_t>(height);
    if (m_impl->frame.bytes() < frameBytes)
    {
        m_impl->frame.allocate(frameBytes);
    }
    if (maskBytes > 0 && m_impl->masks.bytes() < maskBytes)
    {
        m_impl->masks.allocate(maskBytes);
    }

    m_impl->note(cudaMemcpy2DAsync(m_impl->frame.data(), stride,
                                   hostBgr, stride,
                                   static_cast<std::size_t>(width) * 3,
                                   static_cast<std::size_t>(height),
                                   cudaMemcpyHostToDevice, st));

    auto* maskBase = static_cast<std::uint8_t*>(m_impl->masks.data());
    std::size_t off = 0;
    for (const auto& s : segs)
    {
        if (s.mask.data == nullptr || s.mask.width <= 0 || s.mask.height <= 0 ||
            s.mask.channels != 1)
        {
            continue;
        }
        m_impl->note(cudaMemcpy2DAsync(maskBase + off, static_cast<std::size_t>(s.mask.stride),
                                       s.mask.data, static_cast<std::size_t>(s.mask.stride),
                                       static_cast<std::size_t>(s.mask.width),
                                       static_cast<std::size_t>(s.mask.height),
                                       cudaMemcpyHostToDevice, st));

        const int mw = s.mask.width;
        const int mh = s.mask.height;

        // letterbox window of the mask (== computeLetterBox(srcW, srcH, mw, mh))
        const double gain = std::min(static_cast<double>(mh) / height,
                                     static_cast<double>(mw) / width);
        const int newW = static_cast<int>(std::nearbyint(width * gain));
        const int newH = static_cast<int>(std::nearbyint(height * gain));
        const int padL = static_cast<int>(std::nearbyint((mw - newW) / 2.0 - 0.1));
        const int padT = static_cast<int>(std::nearbyint((mh - newH) / 2.0 - 0.1));
        if (newW <= 0 || newH <= 0 || padL < 0 || padT < 0 ||
            padL + newW > mw || padT + newH > mh)
        {
            continue;
        }

        kernels::ops::SegLayer L;
        L.mask       = maskBase + off;
        L.maskW      = mw;
        L.maskH      = mh;
        L.maskStride = static_cast<std::size_t>(s.mask.stride);
        L.newW       = newW;
        L.newH       = newH;
        L.padL       = padL;
        L.padT       = padT;
        L.x0         = padClamp(static_cast<int>(std::lround(s.box.left)) - kBoxPad, 0, width);
        L.y0         = padClamp(static_cast<int>(std::lround(s.box.top)) - kBoxPad, 0, height);
        L.x1         = padClamp(static_cast<int>(std::lround(s.box.right)) + kBoxPad, 0, width);
        L.y1         = padClamp(static_cast<int>(std::lround(s.box.bottom)) + kBoxPad, 0, height);

        const int k = ((s.box.label % 20) + 20) % 20;
        const float a = maskAlpha;
        L.c0 = static_cast<float>(kUltraPalette[k][2]) / 255.0f * a;   // B
        L.c1 = static_cast<float>(kUltraPalette[k][1]) / 255.0f * a;   // G
        L.c2 = static_cast<float>(kUltraPalette[k][0]) / 255.0f * a;   // R

        m_impl->layerHost.push_back(L);
        off += static_cast<std::size_t>(mh) * static_cast<std::size_t>(s.mask.stride);
    }

    const int n = static_cast<int>(m_impl->layerHost.size());
    bool any = false;

    if (n > 0)
    {
        const std::size_t layerBytes =
            sizeof(kernels::ops::SegLayer) * static_cast<std::size_t>(n);
        if (m_impl->layers.bytes() < layerBytes)
        {
            m_impl->layers.allocate(layerBytes);
        }
        m_impl->note(cudaMemcpyAsync(m_impl->layers.data(), m_impl->layerHost.data(),
                                     layerBytes, cudaMemcpyHostToDevice, st));

        const cudaError_t e = kernels::ops::composeSegMasks(
            st, static_cast<std::uint8_t*>(m_impl->frame.data()), width, height, stride,
            static_cast<const kernels::ops::SegLayer*>(m_impl->layers.data()), n, maskAlpha);
        m_impl->note(e);
        any = (e == cudaSuccess);
    }

    // ---- semantic colour map ------------------------------------------------
    if (hasSem)
    {
        const sem::SemanticMask& sm = result.semantics[idx];
        const std::size_t elems = static_cast<std::size_t>(sm.width) *
                                  static_cast<std::size_t>(sm.height);
        if (sm.width == width && sm.height == height && elems > 0)
        {
            if (m_impl->sem.bytes() < elems)
            {
                m_impl->sem.allocate(elems);
            }
            m_impl->note(cudaMemcpyAsync(m_impl->sem.data(), sm.data.data(), elems,
                                         cudaMemcpyHostToDevice, st));
            const cudaError_t e = kernels::ops::composeSemantics(
                st, static_cast<std::uint8_t*>(m_impl->frame.data()), width, height,
                stride, static_cast<const std::uint8_t*>(m_impl->sem.data()), elems, 0.5f);
            m_impl->note(e);
            any = any || (e == cudaSuccess);
        }
    }

    // ---- depth heat map -----------------------------------------------------
    if (hasDepth)
    {
        const depth::DepthMap& dm = result.depths[idx];
        const std::size_t elems = static_cast<std::size_t>(dm.width) *
                                  static_cast<std::size_t>(dm.height);
        if (dm.width == width && dm.height == height && elems > 0)
        {
            float vmin = 0.0f, vmax = 1.0f;
            disparityRange(dm, vmin, vmax);

            if (m_impl->depth.bytes() < elems * sizeof(float))
            {
                m_impl->depth.allocate(elems * sizeof(float));
            }
            m_impl->note(cudaMemcpyAsync(m_impl->depth.data(), dm.data.data(),
                                         elems * sizeof(float),
                                         cudaMemcpyHostToDevice, st));
            const cudaError_t e = kernels::ops::composeDepths(
                st, static_cast<std::uint8_t*>(m_impl->frame.data()), width, height,
                stride, m_impl->depth.asFloat(), elems, vmin, vmax);
            m_impl->note(e);
            any = any || (e == cudaSuccess);
        }
    }

    if (!any)
    {
        return nullptr;
    }

    // The GL backend consumes the frame on its own CUDA stream (the PBO path),
    // so the composite has to be complete before we hand the pointer over.
    m_impl->note(cudaStreamSynchronize(st));

    return static_cast<const std::uint8_t*>(m_impl->frame.data());
}

// -----------------------------------------------------------------------------
//  Host copy-back (for backends that upload from CPU memory)
// -----------------------------------------------------------------------------
const std::uint8_t* OverlayStage::copyBack(int width, int height, std::size_t stride)
{
    if (m_impl->frame.data() == nullptr)
    {
        return nullptr;
    }

    const std::size_t bytes = stride * static_cast<std::size_t>(height);
    if (m_impl->hostOut.bytes() < bytes)
    {
        m_impl->hostOut.allocate(bytes);
    }
    m_impl->note(cudaMemcpy2DAsync(m_impl->hostOut.data(), stride,
                                   m_impl->frame.data(), stride,
                                   static_cast<std::size_t>(width) * 3,
                                   static_cast<std::size_t>(height),
                                   cudaMemcpyDeviceToHost, m_impl->stream));
    m_impl->note(cudaStreamSynchronize(m_impl->stream));

    return static_cast<const std::uint8_t*>(m_impl->hostOut.data());
}

}  // namespace trt_alpha::renderer
