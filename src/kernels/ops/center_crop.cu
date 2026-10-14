// =============================================================================
//  trt_alpha :: kernels :: ops :: center_crop  (implementation)
// -----------------------------------------------------------------------------
//  Host side : the two `int(round(...))` offsets.
//  Device side: a strided copy of the window.
//
//  Mirrors torchvision/transforms/functional.py:592 for a PIL input; see
//  center_crop.hpp for the two details that are easy to get wrong.
// =============================================================================
#include "trt_alpha/kernels/ops/center_crop.hpp"

#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

// -----------------------------------------------------------------------------
//! Python's round() -- half-to-even -- for a non-negative argument.
//!
//! Written out explicitly instead of via nearbyint()/rint() so the result cannot
//! depend on the floating-point rounding mode that happens to be active.
// -----------------------------------------------------------------------------
int roundHalfEven(double x) noexcept
{
    const double f    = std::floor(x);
    const double frac = x - f;
    if (frac > 0.5) return static_cast<int>(f) + 1;
    if (frac < 0.5) return static_cast<int>(f);
    return ((static_cast<long long>(f) & 1LL) == 0) ? static_cast<int>(f)
                                                    : static_cast<int>(f) + 1;
}

// =============================================================================
//  device kernel
// -----------------------------------------------------------------------------
//! One thread = one output pixel (all 3 channels together).
//! Pure data movement: the 32 consecutive ox of a warp read 96 consecutive
//! source bytes, so both the load and the store are coalesced.
__global__ void centerCropKernel(const unsigned char* __restrict__ src,
                                 unsigned char* __restrict__ dst,
                                 int srcW, int srcH,
                                 int left, int top,
                                 int cropW, int cropH)
{
    const int plane = cropW * cropH;
    const int idx   = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                    + static_cast<int>(threadIdx.x);
    if (idx >= plane)
    {
        return;
    }

    const int ox = idx % cropW;
    const int oy = idx / cropW;

    const std::size_t batchStride = static_cast<std::size_t>(srcW) * srcH * 3;
    const std::size_t srcOffset =
        (static_cast<std::size_t>(top + oy) * static_cast<std::size_t>(srcW)
         + static_cast<std::size_t>(left + ox)) * 3;

    const unsigned char* s =
        src + static_cast<std::size_t>(blockIdx.y) * batchStride + srcOffset;
    unsigned char* d =
        dst + static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(plane) * 3
            + static_cast<std::size_t>(idx) * 3;

    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
}

}  // namespace

// =============================================================================
//  computeCenterCrop
// =============================================================================
CenterCropInfo computeCenterCrop(int srcW, int srcH, int cropW, int cropH)
{
    CenterCropInfo info;
    info.srcW  = srcW;
    info.srcH  = srcH;
    info.cropW = cropW;
    info.cropH = cropH;

    if (srcW <= 0 || srcH <= 0 || cropW <= 0 || cropH <= 0)
    {
        return info;   // valid() already false
    }

    // torchvision functional.center_crop (functional.py:592); Python's round()
    // is half-to-even. Only computed for a legal window, so the argument is
    // always non-negative.
    if (cropW <= srcW)
    {
        info.left = roundHalfEven((static_cast<double>(srcW) - cropW) / 2.0);
    }
    if (cropH <= srcH)
    {
        info.top = roundHalfEven((static_cast<double>(srcH) - cropH) / 2.0);
    }

    return info;
}

// =============================================================================
//  CvCenterCropPlan
// =============================================================================
CvCenterCropPlan::CvCenterCropPlan(int srcW, int srcH, int cropW, int cropH)
    : info_(computeCenterCrop(srcW, srcH, cropW, cropH))
{
    // A crop larger than the image would need CenterCrop's zero padding, which
    // is not implemented: fail loudly instead of cropping something that is not
    // there.
    if (!info_.valid())
    {
        err_ = cudaErrorInvalidValue;
    }
}

CvCenterCropPlan::CvCenterCropPlan(CvCenterCropPlan&& other) noexcept
    : info_(other.info_), err_(other.err_)
{
    other.err_ = cudaSuccess;
}

CvCenterCropPlan::~CvCenterCropPlan() = default;

void CvCenterCropPlan::launch(cudaStream_t stream, int batch,
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

    const int  plane = info_.cropW * info_.cropH;
    const dim3 block(kBlock1D, 1);
    const dim3 grid(static_cast<unsigned>((plane + kBlock1D - 1) / kBlock1D),
                    static_cast<unsigned>(batch));

    centerCropKernel<<<grid, block, 0, stream>>>(
        src, dst,
        info_.srcW, info_.srcH,
        info_.left, info_.top,
        info_.cropW, info_.cropH);

    err_ = cudaGetLastError();
}

// =============================================================================
//  centerCropDevice
// =============================================================================
void centerCropDevice(cudaStream_t stream, int batch,
                      const unsigned char* src, unsigned char* dst,
                      int srcW, int srcH, int cropW, int cropH, int cn)
{
    const CvCenterCropPlan plan(srcW, srcH, cropW, cropH);
    if (!plan.ok())
    {
        return;
    }
    plan.launch(stream, batch, src, dst, cn);
}

}  // namespace trt_alpha::kernels::ops
