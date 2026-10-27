// =============================================================================
//  trt_alpha :: kernels :: ops :: overlay  (implementation)
// -----------------------------------------------------------------------------
//  One thread per destination pixel, looping over the instances. The loop is
//  bounded by the box test first (cheap integer compare) so a 1080p frame with
//  ten small detections only touches the pixels those detections cover.
//
//  Sampling is the ATen / cv::INTER_LINEAR half-pixel mapping, evaluated exactly
//  like the host code did:
//      s  = (newLen / srcLen) * (i + 0.5) - 0.5, clamped at 0
//      a  = (int)s, b = min(a + 1, newLen - 1), w1 = s - a, w0 = 1 - w1
//      v  = wy0 * (wx0*v00 + wx1*v01) + wy1 * (wx0*v10 + wx1*v11)
//  and the result is binarised with `> 0.5f`, which is `ops.scale_masks(...)`
//  followed by the `> 0.5` of the python reference.
// =============================================================================
#include "trt_alpha/kernels/ops/overlay.hpp"

#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

namespace trt_alpha::kernels::ops {
namespace {

//! Edge taps for one axis: index0/index1 and their weights.
__device__ __forceinline__ void axisTap(float scale, int i, int len,
                                        int& a, int& b, float& w0, float& w1)
{
    float s = scale * (static_cast<float>(i) + 0.5f) - 0.5f;
    if (s < 0.0f)
    {
        s = 0.0f;
    }
    a  = static_cast<int>(s);
    b  = a + 1;
    if (b > len - 1)
    {
        b = len - 1;
    }
    w1 = s - static_cast<float>(a);
    w0 = 1.0f - w1;
}

__global__ void composeSegMasksKernel(std::uint8_t* __restrict__ bgr,
                                      int width,
                                      int height,
                                      std::size_t stride,
                                      const SegLayer* __restrict__ layers,
                                      int n,
                                      float keepOne)
{
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    std::uint8_t* px = bgr + static_cast<std::size_t>(y) * stride
                     + static_cast<std::size_t>(x) * 3;

    float mcs0 = 0.0f, mcs1 = 0.0f, mcs2 = 0.0f;
    float opa  = 1.0f;
    bool  hit  = false;

    for (int i = 0; i < n; ++i)
    {
        const SegLayer& L = layers[i];
        if (x < L.x0 || x >= L.x1 || y < L.y0 || y >= L.y1)
        {
            continue;
        }

        int   ax, bx, ay, by;
        float wx0, wx1, wy0, wy1;
        axisTap(static_cast<float>(L.newW) / static_cast<float>(width), x, L.newW,
                ax, bx, wx0, wx1);
        axisTap(static_cast<float>(L.newH) / static_cast<float>(height), y, L.newH,
                ay, by, wy0, wy1);

        const std::uint8_t* r0 = L.mask
                               + static_cast<std::size_t>(L.padT + ay) * L.maskStride
                               + static_cast<std::size_t>(L.padL);
        const std::uint8_t* r1 = L.mask
                               + static_cast<std::size_t>(L.padT + by) * L.maskStride
                               + static_cast<std::size_t>(L.padL);

        const float v00 = r0[ax] != 0 ? 1.0f : 0.0f;
        const float v01 = r0[bx] != 0 ? 1.0f : 0.0f;
        const float v10 = r1[ax] != 0 ? 1.0f : 0.0f;
        const float v11 = r1[bx] != 0 ? 1.0f : 0.0f;

        const float v = wy0 * (wx0 * v00 + wx1 * v01) +
                        wy1 * (wx0 * v10 + wx1 * v11);
        if (!(v > 0.5f))
        {
            continue;   // binary mask off: every term of the composite is identity
        }

        mcs0 = fmaxf(mcs0, L.c0);
        mcs1 = fmaxf(mcs1, L.c1);
        mcs2 = fmaxf(mcs2, L.c2);
        opa *= keepOne;
        hit  = true;
    }

    if (!hit)
    {
        return;
    }

    px[0] = static_cast<std::uint8_t>((static_cast<float>(px[0]) / 255.0f * opa + mcs0) * 255.0f);
    px[1] = static_cast<std::uint8_t>((static_cast<float>(px[1]) / 255.0f * opa + mcs1) * 255.0f);
    px[2] = static_cast<std::uint8_t>((static_cast<float>(px[2]) / 255.0f * opa + mcs2) * 255.0f);
}

//! ultralytics Colors(i, bgr=True) -- 20 entries, stored here as BGR so the
//! composite can write channels straight through.
__device__ __constant__ std::uint8_t kPaletteBgr[20][3] = {
    {255,  42,   4}, {235, 219,  11}, {243, 243, 243}, {183, 223,   0},
    {104,  31,  17}, {221, 111, 255}, { 79,  68, 255}, {  0, 237, 204},
    { 68, 243,   0}, {255,   0, 189}, {255, 180,   0}, {186,   0, 221},
    {255, 255,   0}, {  0, 192,  38}, {179, 255,   1}, {255,  36, 125},
    {104,   0, 123}, {108,  27, 255}, { 47, 109, 252}, { 11, 255, 162},
};

//! One JET channel: 255 * clamp(min(4v + a, -4v + b), 0, 1).
__device__ __forceinline__ float jetChan(float v, float a, float b)
{
    float t = fminf(4.0f * v + a, -4.0f * v + b);
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return t * 255.0f;
}

//! OpenCV's COLORMAP_JET, exactly as `colormap_jet` computes it.
__device__ __forceinline__ void jetBgr(int idx, float& b, float& g, float& r)
{
    const float v = static_cast<float>(idx) / 255.0f;
    b = jetChan(v, -1.5f, 4.5f);
    g = jetChan(v, -0.5f, 3.5f);
    r = jetChan(v,  0.5f, 2.5f);
}

__global__ void composeSemanticsKernel(std::uint8_t* __restrict__ bgr,
                                       int width,
                                       int height,
                                       std::size_t stride,
                                       const std::uint8_t* __restrict__ sem,
                                       std::size_t elems,
                                       float alpha,
                                       float keep)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elems)
    {
        return;
    }
    const int x = static_cast<int>(i % static_cast<std::size_t>(width));
    const int y = static_cast<int>(i / static_cast<std::size_t>(width));
    if (y >= height)
    {
        return;
    }

    const int id = sem[i];
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
    if (id != 255)   // ignore_index paints black
    {
        const int k = ((id % 20) + 20) % 20;
        c0 = static_cast<float>(kPaletteBgr[k][0]);
        c1 = static_cast<float>(kPaletteBgr[k][1]);
        c2 = static_cast<float>(kPaletteBgr[k][2]);
    }

    std::uint8_t* p = bgr + static_cast<std::size_t>(y) * stride
                    + static_cast<std::size_t>(x) * 3;
    p[0] = static_cast<std::uint8_t>(__double2int_rn(
        static_cast<double>(static_cast<float>(p[0]) * keep + c0 * alpha)));
    p[1] = static_cast<std::uint8_t>(__double2int_rn(
        static_cast<double>(static_cast<float>(p[1]) * keep + c1 * alpha)));
    p[2] = static_cast<std::uint8_t>(__double2int_rn(
        static_cast<double>(static_cast<float>(p[2]) * keep + c2 * alpha)));
}

__global__ void composeDepthsKernel(std::uint8_t* __restrict__ bgr,
                                    int width,
                                    int height,
                                    std::size_t stride,
                                    const float* __restrict__ depth,
                                    std::size_t elems,
                                    float vmin,
                                    float vmax)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elems)
    {
        return;
    }
    const int x = static_cast<int>(i % static_cast<std::size_t>(width));
    const int y = static_cast<int>(i / static_cast<std::size_t>(width));
    if (y >= height)
    {
        return;
    }

    const float d     = depth[i];
    const bool  valid = d > 0.0f;
    const float val   = valid ? 1.0f / d : 0.0f;
    float dn = (val - vmin) / (vmax - vmin);
    dn = dn < 0.0f ? 0.0f : (dn > 1.0f ? 1.0f : dn);

    // `(dn * 255).astype(uint8)` truncates.
    const int idx = static_cast<int>(dn * 255.0f);

    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f;
    if (valid)
    {
        jetBgr(idx, h0, h1, h2);
    }

    std::uint8_t* p = bgr + static_cast<std::size_t>(y) * stride
                    + static_cast<std::size_t>(x) * 3;
    // addWeighted(im, 1 - 0.6, heat, 0.6, 0)
    p[0] = static_cast<std::uint8_t>(__double2int_rn(
        static_cast<double>(static_cast<float>(p[0]) * 0.4f + h0 * 0.6f)));
    p[1] = static_cast<std::uint8_t>(__double2int_rn(
        static_cast<double>(static_cast<float>(p[1]) * 0.4f + h1 * 0.6f)));
    p[2] = static_cast<std::uint8_t>(__double2int_rn(
        static_cast<double>(static_cast<float>(p[2]) * 0.4f + h2 * 0.6f)));
}

}  // namespace

cudaError_t composeSegMasks(cudaStream_t stream,
                            std::uint8_t* bgr,
                            int width,
                            int height,
                            std::size_t stride,
                            const SegLayer* layers,
                            int n,
                            float alpha)
{
    if (n <= 0 || width <= 0 || height <= 0)
    {
        return cudaSuccess;
    }
    if (bgr == nullptr || layers == nullptr || stride < static_cast<std::size_t>(width) * 3)
    {
        return cudaErrorInvalidValue;
    }

    const dim3 block(kBlock2DX, kBlock2DY);
    const dim3 grid(static_cast<unsigned>((width + kBlock2DX - 1) / kBlock2DX),
                    static_cast<unsigned>((height + kBlock2DY - 1) / kBlock2DY));

    composeSegMasksKernel<<<grid, block, 0, stream>>>(
        bgr, width, height, stride, layers, n, 1.0f - alpha);

    return cudaGetLastError();
}

cudaError_t composeSemantics(cudaStream_t stream,
                             std::uint8_t* bgr,
                             int width,
                             int height,
                             std::size_t stride,
                             const std::uint8_t* sem,
                             std::size_t elems,
                             float alpha)
{
    if (elems == 0 || width <= 0 || height <= 0)
    {
        return cudaSuccess;
    }
    if (bgr == nullptr || sem == nullptr ||
        stride < static_cast<std::size_t>(width) * 3)
    {
        return cudaErrorInvalidValue;
    }

    const unsigned grid =
        static_cast<unsigned>((elems + kBlock1D - 1) / kBlock1D);
    composeSemanticsKernel<<<grid, kBlock1D, 0, stream>>>(
        bgr, width, height, stride, sem, elems, alpha, 1.0f - alpha);

    return cudaGetLastError();
}

cudaError_t composeDepths(cudaStream_t stream,
                          std::uint8_t* bgr,
                          int width,
                          int height,
                          std::size_t stride,
                          const float* depth,
                          std::size_t elems,
                          float vmin,
                          float vmax)
{
    if (elems == 0 || width <= 0 || height <= 0)
    {
        return cudaSuccess;
    }
    if (bgr == nullptr || depth == nullptr ||
        stride < static_cast<std::size_t>(width) * 3)
    {
        return cudaErrorInvalidValue;
    }

    const unsigned grid =
        static_cast<unsigned>((elems + kBlock1D - 1) / kBlock1D);
    composeDepthsKernel<<<grid, kBlock1D, 0, stream>>>(
        bgr, width, height, stride, depth, elems, vmin, vmax);

    return cudaGetLastError();
}

}  // namespace trt_alpha::kernels::ops
