// =============================================================================
//  trt_alpha :: kernels :: ops :: hwc2chw
// -----------------------------------------------------------------------------
//  CUDA implementation of HWC -> CHW.
//
//  Thread mapping for C == 3 (same style as resize):
//      x dimension flattens the H*W pixels of one image
//      y dimension iterates the batch
//      each thread: read src[i*3 .. i*3+2] (contiguous) -> write
//                   dst[c*H*W + i] for c = 0..2
//  Both reads and writes land on contiguous ranges, so no shared memory is needed.
//
//  For C == 1 it degrades to a plain copy using a 1D grid.
// =============================================================================
#include "trt_alpha/kernels/ops/hwc2chw.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

// -----------------------------------------------------------------------------
// C == 3: interleaved HWC -> 3 planes
// -----------------------------------------------------------------------------
template <typename TIn, typename TOut>
__global__ void hwc2chwC3Kernel(
    const TIn* __restrict__ src, TOut* __restrict__ dst,
    int width, int height, int batchSize)
{
    const int pix = width * height;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;   // linear pixel index
    const int b   = blockDim.y * blockIdx.y + threadIdx.y;   // batch index

    if (idx >= pix || b >= batchSize)
    {
        return;
    }

    const TIn* s = src + ((size_t)b * pix + idx) * 3;
    TOut*      d = dst + (size_t)b * pix * 3 + idx;

    d[0]                  = (TOut)s[0];
    d[(size_t)pix]        = (TOut)s[1];
    d[(size_t)pix * 2]    = (TOut)s[2];
}

// -----------------------------------------------------------------------------
// C == 1: HWC and CHW are isomorphic, plain copy
// -----------------------------------------------------------------------------
template <typename TIn, typename TOut>
__global__ void hwc2chwC1Kernel(
    const TIn* __restrict__ src, TOut* __restrict__ dst, long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
    {
        dst[i] = (TOut)src[i];
    }
}

template <typename TIn, typename TOut>
void launchC3(cudaStream_t stream, int batch,
              const TIn* src, TOut* dst, int width, int height)
{
    const int  pix = width * height;
    const dim3 block(kBlock1D, 1);
    const dim3 grid((unsigned)((pix + kBlock1D - 1) / kBlock1D), (unsigned)batch);

    hwc2chwC3Kernel<TIn, TOut><<<grid, block, 0, stream>>>(src, dst, width, height, batch);
}

template <typename TIn, typename TOut>
void launchC1(cudaStream_t stream, const TIn* src, TOut* dst, long long total)
{
    if (total <= 0) return;
    const unsigned grid = (unsigned)((total + kBlock1D - 1) / kBlock1D);
    hwc2chwC1Kernel<TIn, TOut><<<grid, kBlock1D, 0, stream>>>(src, dst, total);
}

}  // namespace

// =============================================================================
//  CvHwcToChwPlan
// =============================================================================
template <typename TIn, typename TOut>
CvHwcToChwPlan<TIn, TOut>::CvHwcToChwPlan(int width, int height, int channels)
    : width_(width), height_(height), channels_(channels)
{
    if (width <= 0 || height <= 0 || !detail::validChannels(channels))
    {
        err_ = cudaErrorInvalidValue;
    }
}

template <typename TIn, typename TOut>
CvHwcToChwPlan<TIn, TOut>::~CvHwcToChwPlan() = default;

template <typename TIn, typename TOut>
CvHwcToChwPlan<TIn, TOut>::CvHwcToChwPlan(CvHwcToChwPlan&& other) noexcept
    : width_(other.width_), height_(other.height_), channels_(other.channels_), err_(other.err_)
{
    other.err_ = cudaSuccess;
}

template <typename TIn, typename TOut>
void CvHwcToChwPlan<TIn, TOut>::launch(cudaStream_t stream, int batch,
                                       const TIn* src, TOut* dst) const
{
    if (err_ != cudaSuccess)                        return;   // construction failed: see lastError()
    if (src == nullptr || dst == nullptr)           { err_ = cudaErrorInvalidValue;   return; }
    if (batch <= 0)                                 return;
    if (batch > 65535)                              { err_ = cudaErrorInvalidValue;   return; }
    if ((long long)width_ * height_ > 0x7fffffffLL) { err_ = cudaErrorInvalidValue;   return; }

    if (channels_ == 3)
    {
        launchC3<TIn, TOut>(stream, batch, src, dst, width_, height_);
    }
    else
    {
        launchC1<TIn, TOut>(stream, src, dst, (long long)batch * width_ * height_);
    }

    // Fetch the error only, no sync (stay asynchronous)
    err_ = cudaGetLastError();
}

// =============================================================================
//  hwc2chwDevice -- one-shot convenience version
// =============================================================================
template <typename TIn, typename TOut>
void hwc2chwDevice(cudaStream_t stream, int batch,
                   const TIn* src, TOut* dst,
                   int width, int height, int channels)
{
    CvHwcToChwPlan<TIn, TOut> plan(width, height, channels);
    plan.launch(stream, batch, src, dst);
}

// =============================================================================
//  Explicit instantiation
// =============================================================================
template class CvHwcToChwPlan<unsigned char, unsigned char>;
template class CvHwcToChwPlan<float, float>;

template void hwc2chwDevice<unsigned char, unsigned char>(
    cudaStream_t, int, const unsigned char*, unsigned char*, int, int, int);
template void hwc2chwDevice<float, float>(
    cudaStream_t, int, const float*, float*, int, int, int);

}  // namespace trt_alpha::kernels::ops
