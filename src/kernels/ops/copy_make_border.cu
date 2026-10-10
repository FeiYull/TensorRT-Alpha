// =============================================================================
//  trt_alpha :: kernels :: ops :: copy_make_border
// -----------------------------------------------------------------------------
//  CUDA implementation of BORDER_CONSTANT.
//
//  Equivalence with OpenCV (copy.cpp):
//      dst.setTo(value) + src.copyTo(dst(Rect(left,top,w,h)))
//  ==  pixel by pixel: the centre region takes src, everything else takes value
//
//  There is no precomputed table and no device buffer, so the kernel is the only
//  entity.
// =============================================================================
#include "trt_alpha/kernels/ops/copy_make_border.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

// -----------------------------------------------------------------------------
// BORDER_CONSTANT: each thread handles one dst pixel (including its CN channels)
//
// Thread mapping (same style as resize):
//   x dimension (blockIdx.x/threadIdx.x) flattens the dstW*dstH of one image
//   y dimension (blockIdx.y/threadIdx.y) iterates the batch index
// -----------------------------------------------------------------------------
template <typename TIn, typename TOut, int CN>
__global__ void copyMakeBorderConstantKernel(
    const TIn* __restrict__ src, int srcW, int srcH,
    TOut* __restrict__ dst, int dstW, int dstH,
    int top, int left, int batchSize,
    TOut f0, TOut f1, TOut f2)
{
    const int pix = dstW * dstH;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;   // linear dst pixel index
    const int b   = blockDim.y * blockIdx.y + threadIdx.y;   // batch index

    if (idx >= pix || b >= batchSize)
    {
        return;
    }

    const int y = idx / dstW;
    const int x = idx - y * dstW;

    // Map back to source coordinates: only [0,srcW) x [0,srcH) is the "centre region"
    const int sy = y - top;
    const int sx = x - left;

    const TIn* srcImg = src + (size_t)b * srcW * srcH * CN;
    TOut*      out    = dst + ((size_t)b * pix + idx) * CN;

    if (sy >= 0 && sy < srcH && sx >= 0 && sx < srcW)
    {
        const TIn* s = srcImg + ((size_t)sy * srcW + sx) * CN;
#pragma unroll
        for (int c = 0; c < CN; ++c)
        {
            out[c] = (TOut)s[c];
        }
    }
    else
    {
        if constexpr (CN >= 1) out[0] = f0;
        if constexpr (CN >= 2) out[1] = f1;
        if constexpr (CN >= 3) out[2] = f2;
    }
}

template <typename TIn, typename TOut, int CN>
void launchOne(cudaStream_t stream, int batch,
               const TIn* src, TOut* dst,
               int srcW, int srcH, int dstW, int dstH,
               int top, int left,
               TOut f0, TOut f1, TOut f2)
{
    const int  pix = dstW * dstH;
    const dim3 block(kBlock1D, 1);
    const dim3 grid((unsigned)((pix + kBlock1D - 1) / kBlock1D), (unsigned)batch);

    copyMakeBorderConstantKernel<TIn, TOut, CN><<<grid, block, 0, stream>>>(
        src, srcW, srcH, dst, dstW, dstH, top, left, batch, f0, f1, f2);
}

}  // namespace

// =============================================================================
//  CvCopyMakeBorderPlan
// =============================================================================
template <typename TIn, typename TOut>
CvCopyMakeBorderPlan<TIn, TOut>::CvCopyMakeBorderPlan(
    int srcW, int srcH, int top, int bottom, int left, int right)
    : srcW_(srcW), srcH_(srcH), top_(top), bottom_(bottom), left_(left), right_(right)
{
    if (srcW <= 0 || srcH <= 0 || top < 0 || bottom < 0 || left < 0 || right < 0)
    {
        err_ = cudaErrorInvalidValue;
    }
}

template <typename TIn, typename TOut>
CvCopyMakeBorderPlan<TIn, TOut>::~CvCopyMakeBorderPlan() = default;

template <typename TIn, typename TOut>
CvCopyMakeBorderPlan<TIn, TOut>::CvCopyMakeBorderPlan(CvCopyMakeBorderPlan&& other) noexcept
    : srcW_(other.srcW_), srcH_(other.srcH_), top_(other.top_), bottom_(other.bottom_),
      left_(other.left_), right_(other.right_), err_(other.err_)
{
    other.err_ = cudaSuccess;
}

template <typename TIn, typename TOut>
void CvCopyMakeBorderPlan<TIn, TOut>::launch(
    cudaStream_t stream, int batch,
    const TIn* src, TOut* dst, int cn,
    BorderType borderType, const double* value) const
{
    if (err_ != cudaSuccess)                      return;   // construction failed: see lastError()
    if (borderType != BorderType::Constant)       { err_ = cudaErrorNotSupported; return; }
    if (src == nullptr || dst == nullptr)         { err_ = cudaErrorInvalidValue;   return; }
    if (!detail::validChannels(cn))               { err_ = cudaErrorInvalidValue;   return; }
    if (batch <= 0)                               return;
    if (batch > 65535)                            { err_ = cudaErrorInvalidValue;   return; }

    const int dstW = srcW_ + left_ + right_;
    const int dstH = srcH_ + top_ + bottom_;
    if ((long long)dstW * dstH > 0x7fffffffLL)    { err_ = cudaErrorInvalidValue;   return; }

    // Fill value: converted once on the host side to the target type (one per
    // channel; missing channels fall back to channel 0)
    const double v0 = value ? value[0] : 0.0;
    const double v1 = value ? (cn >= 2 ? value[1] : value[0]) : 0.0;
    const double v2 = value ? (cn >= 3 ? value[2] : value[0]) : 0.0;
    const TOut f0 = detail::CastFill<TOut>::make(v0);
    const TOut f1 = detail::CastFill<TOut>::make(v1);
    const TOut f2 = detail::CastFill<TOut>::make(v2);

    switch (cn)
    {
        case 1:
            launchOne<TIn, TOut, 1>(stream, batch, src, dst,
                                    srcW_, srcH_, dstW, dstH, top_, left_, f0, f1, f2);
            break;
        case 3:
            launchOne<TIn, TOut, 3>(stream, batch, src, dst,
                                    srcW_, srcH_, dstW, dstH, top_, left_, f0, f1, f2);
            break;
        default:
            break;
    }

    // Fetch the error only, no sync (stay asynchronous)
    err_ = cudaGetLastError();
}

// =============================================================================
//  copyMakeBorderDevice -- one-shot convenience version
// =============================================================================
template <typename TIn, typename TOut>
void copyMakeBorderDevice(cudaStream_t stream, int batch,
                          const TIn* src, TOut* dst,
                          int srcW, int srcH, int cn,
                          int top, int bottom, int left, int right,
                          BorderType borderType, const double* value)
{
    CvCopyMakeBorderPlan<TIn, TOut> plan(srcW, srcH, top, bottom, left, right);
    plan.launch(stream, batch, src, dst, cn, borderType, value);
}

// =============================================================================
//  Explicit instantiation
// =============================================================================
template class CvCopyMakeBorderPlan<unsigned char, unsigned char>;
template class CvCopyMakeBorderPlan<float, float>;

template void copyMakeBorderDevice<unsigned char, unsigned char>(
    cudaStream_t, int, const unsigned char*, unsigned char*, int, int, int,
    int, int, int, int, BorderType, const double*);
template void copyMakeBorderDevice<float, float>(
    cudaStream_t, int, const float*, float*, int, int, int,
    int, int, int, int, BorderType, const double*);

}  // namespace trt_alpha::kernels::ops
