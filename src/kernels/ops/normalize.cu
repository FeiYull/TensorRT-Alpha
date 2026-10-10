// =============================================================================
//  trt_alpha :: kernels :: ops :: normalize
// -----------------------------------------------------------------------------
//  Element-wise linear scaling. One kernel instantiation per mode (template
//  parameter bool kAffine).
//
//  Precision-critical points:
//      DivConst uses __fdiv_rn -- IEEE correctly-rounded float division, bit-exact
//      with (float)x / 255.0f.
//      Never write __fmul_rn(x, 1.0f/255.0f): that rounds 1/255 first, and 126 of
//      the 256 measured values end up 1 ULP off (see the measurements at the top of
//      normalize.hpp).
//
//      Affine uses __fmul_rn + __fadd_rn separately to stop nvcc from contracting
//      mul+add into an FMA -- same ordering as cv::Mat::convertTo's "multiply then
//      add" (resize's lerp2 works the same way).
// =============================================================================
#include "trt_alpha/kernels/ops/normalize.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

// -----------------------------------------------------------------------------
// Element-wise: dst[i] = src[i] / a      (kAffine == false)
//               dst[i] = src[i] * a + b  (kAffine == true)
// -----------------------------------------------------------------------------
template <typename TIn, typename TOut, bool kAffine>
__global__ void normalizeKernel(
    const TIn* __restrict__ src, TOut* __restrict__ dst,
    long long total, float a, float b)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
    {
        return;
    }

    const float x = (float)src[i];   // uchar -> float is exact (no rounding)

    float y;
    if constexpr (kAffine)
    {
        y = __fadd_rn(__fmul_rn(x, a), b);
    }
    else
    {
        y = __fdiv_rn(x, a);         // a is the divisor here
    }

    dst[i] = (TOut)y;
}

template <typename TIn, typename TOut, bool kAffine>
void launchOne(cudaStream_t stream, const TIn* src, TOut* dst,
               long long total, float a, float b)
{
    if (total <= 0)
    {
        return;
    }
    const long long blocks = (total + kBlock1D - 1) / kBlock1D;
    const unsigned  grid   = (unsigned)(blocks > 0x7fffffffLL ? 0x7fffffffLL : blocks);

    normalizeKernel<TIn, TOut, kAffine><<<grid, kBlock1D, 0, stream>>>(src, dst, total, a, b);
}

}  // namespace

// =============================================================================
//  CvNormalizePlan
// =============================================================================
template <typename TIn, typename TOut>
CvNormalizePlan<TIn, TOut>::CvNormalizePlan() = default;   // DivConst / 255

template <typename TIn, typename TOut>
CvNormalizePlan<TIn, TOut> CvNormalizePlan<TIn, TOut>::byDivisor(float divisor)
{
    CvNormalizePlan p;
    p.mode_ = NormMode::DivConst;
    p.a_    = divisor;
    p.b_    = 0.0f;
    if (!(divisor != 0.0f))
    {
        p.err_ = cudaErrorInvalidValue;   // division by zero
    }
    return p;
}

template <typename TIn, typename TOut>
CvNormalizePlan<TIn, TOut> CvNormalizePlan<TIn, TOut>::byAffine(float alpha, float beta)
{
    CvNormalizePlan p;
    p.mode_ = NormMode::Affine;
    p.a_    = alpha;
    p.b_    = beta;
    return p;
}

template <typename TIn, typename TOut>
CvNormalizePlan<TIn, TOut>::~CvNormalizePlan() = default;

template <typename TIn, typename TOut>
CvNormalizePlan<TIn, TOut>::CvNormalizePlan(CvNormalizePlan&& other) noexcept
    : mode_(other.mode_), a_(other.a_), b_(other.b_), err_(other.err_)
{
    other.err_ = cudaSuccess;
}

template <typename TIn, typename TOut>
void CvNormalizePlan<TIn, TOut>::launch(cudaStream_t stream, int batch,
                                        const TIn* src, TOut* dst,
                                        int cn, int hw) const
{
    if (err_ != cudaSuccess)              return;   // construction failed: see lastError()
    if (src == nullptr || dst == nullptr) { err_ = cudaErrorInvalidValue; return; }
    if (!detail::validChannels(cn))       { err_ = cudaErrorInvalidValue; return; }
    if (hw <= 0)                          { err_ = cudaErrorInvalidValue; return; }
    if (batch <= 0)                       return;

    const long long total = (long long)batch * cn * hw;

    if (mode_ == NormMode::Affine)
    {
        launchOne<TIn, TOut, true>(stream, src, dst, total, a_, b_);
    }
    else
    {
        launchOne<TIn, TOut, false>(stream, src, dst, total, a_, b_);
    }

    // Fetch the error only, no sync (stay asynchronous)
    err_ = cudaGetLastError();
}

// =============================================================================
//  normalizeDevice -- one-shot convenience version
// =============================================================================
template <typename TIn, typename TOut>
void normalizeDevice(cudaStream_t stream, int batch,
                     const TIn* src, TOut* dst,
                     int cn, int hw,
                     NormMode mode, float a, float b)
{
    CvNormalizePlan<TIn, TOut> plan = (mode == NormMode::Affine)
                                        ? CvNormalizePlan<TIn, TOut>::byAffine(a, b)
                                        : CvNormalizePlan<TIn, TOut>::byDivisor(a);
    plan.launch(stream, batch, src, dst, cn, hw);
}

// =============================================================================
//  Explicit instantiation
// =============================================================================
template class CvNormalizePlan<unsigned char, float>;
template class CvNormalizePlan<float, float>;

template void normalizeDevice<unsigned char, float>(
    cudaStream_t, int, const unsigned char*, float*, int, int, NormMode, float, float);
template void normalizeDevice<float, float>(
    cudaStream_t, int, const float*, float*, int, int, NormMode, float, float);

}  // namespace trt_alpha::kernels::ops
