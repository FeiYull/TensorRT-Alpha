// =============================================================================
//  trt_alpha :: kernels :: ops :: pil_resample  (implementation)
// -----------------------------------------------------------------------------
//  Host side : the torchvision "shortest edge" geometry and the two fixed-point
//              coefficient tables, plus their upload.
//  Device side: one kernel reproducing Pillow's resize in a single pass.
//
//  Every formula below is a line-by-line port of
//      pillow-12.3.0/src/libImaging/Resample.c
//  and
//      torchvision/transforms/{transforms.py,_functional_pil.py}
//  with the exact versions this project validates against
//      torchvision 0.29.0 / Pillow 12.3.0 / torch 2.14.0
//  Line references in the comments point at the reference sources so the port can
//  be re-checked without guessing.
// =============================================================================
#include "trt_alpha/kernels/ops/pil_resample.hpp"

#include "pil_resample_detail.cuh"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_alpha::kernels::ops {
namespace {

//! Pillow's `static struct filter BILINEAR = {bilinear_filter, 1.0};`
//! (Resample.c:84) -- a triangle kernel with support 1.0.
constexpr double kPilBilinearSupport = 1.0;

// =============================================================================
//  buildPilResampleAxis
// -----------------------------------------------------------------------------
//! Mirror of Pillow's precompute_coeffs + normalize_coeffs_8bpc for the BILINEAR
//! filter, with box = (0, inSize) as Image.resize() always passes.
// =============================================================================
PilResampleAxis buildPilResampleAxis(int inSize, int outSize)
{
    PilResampleAxis ax;
    if (inSize <= 0 || outSize <= 0)
    {
        return ax;
    }

    // double scale = (double)(in1 - in0) / outSize;      (Resample.c:198)
    // in0 / in1 come from Pillow's `float box[4]`, so the subtraction is a FLOAT
    // operation promoted to double; with in0 = 0 that is just (float)inSize.
    const float  span  = static_cast<float>(inSize) - 0.0f;
    const double scale = static_cast<double>(span) / static_cast<double>(outSize);

    double filterscale = scale;
    if (filterscale < 1.0)
    {
        filterscale = 1.0;        // <- only reachable when upscaling
    }

    // support = filterp->support * filterscale;   ksize = ceil(support)*2 + 1
    // Enlarging the support with the scale is exactly what makes Pillow
    // antialias when downscaling, and it is the reason cv2's INTER_LINEAR cannot
    // be used as a substitute here.
    const double support = kPilBilinearSupport * filterscale;
    const int    ksize   = static_cast<int>(std::ceil(support)) * 2 + 1;

    ax.ksize = ksize;
    ax.bounds.assign(static_cast<std::size_t>(outSize), PilBound{});

    // Pillow keeps the weights in `double` and only quantizes at the very end,
    // so the intermediate buffer has to be double as well.
    std::vector<double> dk(static_cast<std::size_t>(outSize) * static_cast<std::size_t>(ksize), 0.0);

    const double inv_filterscale = 1.0 / filterscale;   // invariant over the loop

    for (int xx = 0; xx < outSize; ++xx)
    {
        const double center = 0.0 + (xx + 0.5) * scale;   // in0 = 0
        double       ww     = 0.0;

        // (int) truncates toward zero -- for a negative argument that is NOT
        // floor, which is why the xmin < 0 clamp below is load-bearing.
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0)
        {
            xmin = 0;
        }
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > inSize)
        {
            xmax = inSize;
        }
        const int count = xmax - xmin;

        double* k = dk.data() + static_cast<std::size_t>(xx) * static_cast<std::size_t>(ksize);
        for (int x = 0; x < count; ++x)
        {
            // bilinear_filter() (Resample.c:21)
            double t = (x + xmin - center + 0.5) * inv_filterscale;
            if (t < 0.0)
            {
                t = -t;
            }
            const double w = (t < 1.0) ? (1.0 - t) : 0.0;
            k[x] = w;
            ww += w;
        }
        if (ww != 0.0)
        {
            for (int x = 0; x < count; ++x)
            {
                k[x] /= ww;       // normalize so the taps sum to 1
            }
        }
        // Entries past `count` keep the 0 they were initialized with, matching
        // Pillow's explicit zero-fill of the unused tail.

        ax.bounds[static_cast<std::size_t>(xx)] = PilBound{xmin, count};
    }

    // normalize_coeffs_8bpc (Resample.c:269): quantize to INT32 fixed point.
    //      k < 0 -> (int)(-0.5 + k * (1 << 22))
    //      else  -> (int)( 0.5 + k * (1 << 22))
    // The multiply is by a power of two and therefore exact, so contraction into
    // an FMA cannot change the result -- the only rounding is the truncation.
    ax.coef.resize(dk.size());
    for (std::size_t i = 0; i < dk.size(); ++i)
    {
        const double v = dk[i] * static_cast<double>(kPilCoefScale);
        ax.coef[i] = (v < 0.0) ? static_cast<std::int32_t>(v - 0.5)
                               : static_cast<std::int32_t>(v + 0.5);
    }

    return ax;
}

// =============================================================================
//  device kernel
// -----------------------------------------------------------------------------
//! One thread = one output pixel (all 3 channels together).
//!
//! Pillow's two passes are folded into one: for every vertical tap row the
//! horizontal value is computed (including ITS clip8, exactly like Pillow's
//! uint8 intermediate) and then accumulated with the vertical coefficient. The
//! integers and the clip8 calls are identical, so the result is identical --
//! without materializing the intermediate buffer.
//!
//! Writes: the dx of 32 threads within a warp are consecutive, so each 3-byte
//! store is coalesced.
//! Reads : each tap row is read from L1/L2; neighbouring output pixels share
//! their source window whenever the image is downscaled.
__global__ void pilResampleKernel(const unsigned char* __restrict__ src,
                                  unsigned char* __restrict__ dst,
                                  int srcW, int srcH,
                                  int dstW, int dstH,
                                  int needH, int needV,
                                  const PilBound* __restrict__ bH,
                                  const int* __restrict__ kH, int ksizeH,
                                  const PilBound* __restrict__ bV,
                                  const int* __restrict__ kV, int ksizeV)
{
    const int plane = dstW * dstH;
    const int idx   = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                    + static_cast<int>(threadIdx.x);
    if (idx >= plane)
    {
        return;
    }

    const int dx = idx % dstW;                 // column in the resized image
    const int dy = idx / dstW;                 // row    in the resized image

    const unsigned char* im =
        src + static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(srcW) * srcH * 3;

    unsigned char o0, o1, o2;
    if (needV)
    {
        // _ImagingResampleVertical_8bpc, bands == 3 (Resample.c:446).
        // The horizontal pass is row preserving, so resized row r is built from
        // source row r -- no temporary row offset is involved.
        const PilBound bv = bV[dy];
        const int*     kv = kV + static_cast<std::size_t>(dy) * ksizeV;

        int s0 = kPilRoundHalf, s1 = kPilRoundHalf, s2 = kPilRoundHalf;
        for (int t = 0; t < bv.count; ++t)
        {
            const unsigned char* row =
                im + static_cast<std::size_t>(bv.first + t) * static_cast<std::size_t>(srcW) * 3;
            const uchar3 h  = detail::pilHorizPixel(row, dx, needH, bH, kH, ksizeH);
            const int    kt = kv[t];
            s0 += h.x * kt;
            s1 += h.y * kt;
            s2 += h.z * kt;
        }
        o0 = detail::pilClip8(s0);
        o1 = detail::pilClip8(s1);
        o2 = detail::pilClip8(s2);
    }
    else
    {
        // The vertical pass is skipped: the resized image IS the horizontally
        // resampled one, so the pixel is taken as is (still possibly filtered
        // horizontally, still clipped there).
        const unsigned char* row = im + static_cast<std::size_t>(dy) * static_cast<std::size_t>(srcW) * 3;
        const uchar3         h   = detail::pilHorizPixel(row, dx, needH, bH, kH, ksizeH);
        o0 = h.x;
        o1 = h.y;
        o2 = h.z;
    }

    unsigned char* o = dst + static_cast<std::size_t>(blockIdx.y)
                            * static_cast<std::size_t>(plane) * 3
                            + static_cast<std::size_t>(idx) * 3;
    o[0] = o0;
    o[1] = o1;
    o[2] = o2;
}

}  // namespace

// =============================================================================
//  computePilResizeShortestEdge
// =============================================================================
void computePilResizeShortestEdge(int srcW, int srcH, int size, int& newW, int& newH)
{
    if (srcW <= 0 || srcH <= 0 || size <= 0)
    {
        newW = 0;
        newH = 0;
        return;
    }

    const bool wIsShort  = (srcW <= srcH);
    const int  shortEdge = wIsShort ? srcW : srcH;
    const int  longEdge  = wIsShort ? srcH : srcW;

    // The product is taken in 64-bit so the division sees the same numerator
    // Python computes exactly before its single float division.
    const long long num = static_cast<long long>(size) * static_cast<long long>(longEdge);
    const int newLong   = static_cast<int>(static_cast<double>(num)
                                           / static_cast<double>(shortEdge));

    newW = wIsShort ? size : newLong;
    newH = wIsShort ? newLong : size;
}

// =============================================================================
//  CvPilResamplePlan
// =============================================================================
CvPilResamplePlan::CvPilResamplePlan(int srcW, int srcH, int dstW, int dstH)
    : srcW_(srcW), srcH_(srcH), dstW_(dstW), dstH_(dstH)
{
    if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    // Pillow skips a pass whose output extent equals its input extent
    // (Resample.c:791-792 with box = (0, 0, W, H)).
    needH_ = (dstW != srcW);
    needV_ = (dstH != srcH);

    // The vertical table is always built: even when the pass is skipped its
    // geometry is what tells the kernel how many rows the output has.
    axisV_ = buildPilResampleAxis(srcH, dstH);
    if (needH_)
    {
        axisH_ = buildPilResampleAxis(srcW, dstW);
    }

    // ---- upload the coefficient tables --------------------------------------
    cudaError_t e = cudaSuccess;
    if (needH_)
    {
        if (axisH_.bounds.empty() || axisH_.coef.empty() || axisH_.ksize <= 0)
        {
            e = cudaErrorInvalidValue;
        }
        else
        {
            e = cudaMalloc(reinterpret_cast<void**>(&d_boundsH_),
                           axisH_.bounds.size() * sizeof(PilBound));
            if (e == cudaSuccess)
            {
                e = cudaMemcpy(d_boundsH_, axisH_.bounds.data(),
                               axisH_.bounds.size() * sizeof(PilBound),
                               cudaMemcpyHostToDevice);
            }
        }
        if (e == cudaSuccess)
        {
            e = cudaMalloc(reinterpret_cast<void**>(&d_coefH_),
                           axisH_.coef.size() * sizeof(std::int32_t));
            if (e == cudaSuccess)
            {
                e = cudaMemcpy(d_coefH_, axisH_.coef.data(),
                               axisH_.coef.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice);
            }
        }
    }

    if (e == cudaSuccess)
    {
        if (axisV_.bounds.empty() || axisV_.coef.empty() || axisV_.ksize <= 0)
        {
            e = cudaErrorInvalidValue;
        }
        else
        {
            e = cudaMalloc(reinterpret_cast<void**>(&d_boundsV_),
                           axisV_.bounds.size() * sizeof(PilBound));
            if (e == cudaSuccess)
            {
                e = cudaMemcpy(d_boundsV_, axisV_.bounds.data(),
                               axisV_.bounds.size() * sizeof(PilBound),
                               cudaMemcpyHostToDevice);
            }
        }
        if (e == cudaSuccess)
        {
            e = cudaMalloc(reinterpret_cast<void**>(&d_coefV_),
                           axisV_.coef.size() * sizeof(std::int32_t));
            if (e == cudaSuccess)
            {
                e = cudaMemcpy(d_coefV_, axisV_.coef.data(),
                               axisV_.coef.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice);
            }
        }
    }

    if (e != cudaSuccess)
    {
        err_ = e;
        TRT_LOG_ERROR("pil_resample: coefficient table upload failed: "
                      + std::string(cudaGetErrorString(e)));
        release();
    }
}

CvPilResamplePlan::CvPilResamplePlan(CvPilResamplePlan&& other) noexcept
    : srcW_(other.srcW_),
      srcH_(other.srcH_),
      dstW_(other.dstW_),
      dstH_(other.dstH_),
      needH_(other.needH_),
      needV_(other.needV_),
      axisH_(std::move(other.axisH_)),
      axisV_(std::move(other.axisV_)),
      d_boundsH_(other.d_boundsH_),
      d_coefH_(other.d_coefH_),
      d_boundsV_(other.d_boundsV_),
      d_coefV_(other.d_coefV_),
      err_(other.err_)
{
    other.d_boundsH_ = nullptr;
    other.d_coefH_   = nullptr;
    other.d_boundsV_ = nullptr;
    other.d_coefV_   = nullptr;
    other.err_       = cudaSuccess;
}

void CvPilResamplePlan::release() noexcept
{
    if (d_boundsH_ != nullptr) { cudaFree(d_boundsH_); d_boundsH_ = nullptr; }
    if (d_coefH_   != nullptr) { cudaFree(d_coefH_);   d_coefH_   = nullptr; }
    if (d_boundsV_ != nullptr) { cudaFree(d_boundsV_); d_boundsV_ = nullptr; }
    if (d_coefV_   != nullptr) { cudaFree(d_coefV_);   d_coefV_   = nullptr; }
}

CvPilResamplePlan::~CvPilResamplePlan()
{
    release();
}

void CvPilResamplePlan::launch(cudaStream_t stream, int batch,
                               const unsigned char* src, unsigned char* dst, int cn) const
{
    if (err_ != cudaSuccess || batch <= 0 || src == nullptr || dst == nullptr)
    {
        return;
    }
    if (cn != 3)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    const int  plane = dstW_ * dstH_;
    const dim3 block(kBlock1D, 1);
    const dim3 grid(static_cast<unsigned>((plane + kBlock1D - 1) / kBlock1D),
                    static_cast<unsigned>(batch));

    pilResampleKernel<<<grid, block, 0, stream>>>(
        src, dst,
        srcW_, srcH_, dstW_, dstH_,
        needH_ ? 1 : 0, needV_ ? 1 : 0,
        d_boundsH_, d_coefH_, axisH_.ksize,
        d_boundsV_, d_coefV_, axisV_.ksize);

    err_ = cudaGetLastError();
}

// =============================================================================
//  pilResampleDevice
// =============================================================================
void pilResampleDevice(cudaStream_t stream, int batch,
                       const unsigned char* src, unsigned char* dst,
                       int srcW, int srcH, int dstW, int dstH, int cn)
{
    const CvPilResamplePlan plan(srcW, srcH, dstW, dstH);
    if (!plan.ok())
    {
        TRT_LOG_ERROR("pil_resample: one-shot plan is unusable");
        return;
    }
    plan.launch(stream, batch, src, dst, cn);
}

}  // namespace trt_alpha::kernels::ops
