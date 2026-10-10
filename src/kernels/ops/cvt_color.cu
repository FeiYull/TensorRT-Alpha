// =============================================================================
//  trt_alpha :: kernels :: ops :: cvt_color
// -----------------------------------------------------------------------------
//  CUDA implementation of COLOR_BGR2RGB: dst = {src[2], src[1], src[0]}
//  (pixel by pixel). Bit-exact with OpenCV's swapBlue / mixChannels(fromTo={2,1,0})
//  since it is a pure assignment.
// =============================================================================
#include "trt_alpha/kernels/ops/cvt_color.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

// -----------------------------------------------------------------------------
// Each thread handles one pixel (3 channels)
//
// Thread mapping (same style as resize):
//   x dimension flattens the width*height of one image
//   y dimension iterates the batch
//
// Access: thread idx reads src[idx*3 .. idx*3+2]; neighbouring threads read
// neighbouring triples -> the whole warp covers a contiguous range, so both reads
// and writes are coalesced.
// -----------------------------------------------------------------------------
template <typename TIn, typename TOut>
__global__ void cvtColorBgr2RgbKernel(
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
    TOut*      d = dst + ((size_t)b * pix + idx) * 3;

    d[0] = (TOut)s[2];   // R <- B
    d[1] = (TOut)s[1];   // G <- G
    d[2] = (TOut)s[0];   // B <- R
}

template <typename TIn, typename TOut>
void launchOne(cudaStream_t stream, int batch,
               const TIn* src, TOut* dst, int width, int height)
{
    const int  pix = width * height;
    const dim3 block(kBlock1D, 1);
    const dim3 grid((unsigned)((pix + kBlock1D - 1) / kBlock1D), (unsigned)batch);

    cvtColorBgr2RgbKernel<TIn, TOut><<<grid, block, 0, stream>>>(
        src, dst, width, height, batch);
}

}  // namespace

// =============================================================================
//  CvCvtColorPlan
// =============================================================================
template <typename TIn, typename TOut>
CvCvtColorPlan<TIn, TOut>::CvCvtColorPlan(int width, int height, ColorCode code)
    : width_(width), height_(height), code_(code)
{
    if (width <= 0 || height <= 0)
    {
        err_ = cudaErrorInvalidValue;
    }
}

template <typename TIn, typename TOut>
CvCvtColorPlan<TIn, TOut>::~CvCvtColorPlan() = default;

template <typename TIn, typename TOut>
CvCvtColorPlan<TIn, TOut>::CvCvtColorPlan(CvCvtColorPlan&& other) noexcept
    : width_(other.width_), height_(other.height_), code_(other.code_), err_(other.err_)
{
    other.err_ = cudaSuccess;
}

template <typename TIn, typename TOut>
void CvCvtColorPlan<TIn, TOut>::launch(cudaStream_t stream, int batch,
                                       const TIn* src, TOut* dst, int cn) const
{
    if (err_ != cudaSuccess)                        return;   // construction failed: see lastError()
    if (code_ != ColorCode::BGR2RGB)                { err_ = cudaErrorNotSupported; return; }
    if (src == nullptr || dst == nullptr)           { err_ = cudaErrorInvalidValue;   return; }
    if (cn != 3)                                    { err_ = cudaErrorInvalidValue;   return; }
    if (batch <= 0)                                 return;
    if (batch > 65535)                              { err_ = cudaErrorInvalidValue;   return; }
    if ((long long)width_ * height_ > 0x7fffffffLL) { err_ = cudaErrorInvalidValue;   return; }

    launchOne<TIn, TOut>(stream, batch, src, dst, width_, height_);

    // Fetch the error only, no sync (stay asynchronous)
    err_ = cudaGetLastError();
}

// =============================================================================
//  cvtColorDevice -- one-shot convenience version
// =============================================================================
template <typename TIn, typename TOut>
void cvtColorDevice(cudaStream_t stream, int batch,
                    const TIn* src, TOut* dst,
                    int width, int height, int cn, ColorCode code)
{
    CvCvtColorPlan<TIn, TOut> plan(width, height, code);
    plan.launch(stream, batch, src, dst, cn);
}

// =============================================================================
//  Explicit instantiation
// =============================================================================
template class CvCvtColorPlan<unsigned char, unsigned char>;
template class CvCvtColorPlan<float, float>;

template void cvtColorDevice<unsigned char, unsigned char>(
    cudaStream_t, int, const unsigned char*, unsigned char*, int, int, int, ColorCode);
template void cvtColorDevice<float, float>(
    cudaStream_t, int, const float*, float*, int, int, int, ColorCode);

}  // namespace trt_alpha::kernels::ops
