// =============================================================================
//  trt_alpha :: kernels :: ops :: seg_mask_post  (implementation)
// -----------------------------------------------------------------------------
//  Grid shape for binarizeCropMasks:
//      block = (kBlock1D = 256)          one row of the plane per warp-group pass
//      grid  = (netH, n)                 x -> row index, y -> instance index
//
//  Rows, not a flat index, so the innermost loop never divides by netW. Both the
//  load and the store stay coalesced because a row of the plane is contiguous.
//
//  The per-instance foreground count is reduced inside each block and folded with
//  one atomicAdd per block, so `counts` costs nothing measurable next to the
//  plane it walks.
// =============================================================================
#include "trt_alpha/kernels/ops/seg_mask_post.hpp"

#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

//! Same arithmetic as the host used to do: ceil() in double, then clamp to the
//! plane. Kept in double on purpose -- float ceil() disagrees on exact integers
//! that are not representable in binary32.
__device__ __forceinline__ int ceilClamp(double v, int hi)
{
    const int c = static_cast<int>(ceil(v));
    return c < 0 ? 0 : (c > hi ? hi : c);
}

__global__ void binarizeCropMaskKernel(float* __restrict__ up,
                                       std::size_t upStep,
                                       const float* __restrict__ boxes,
                                       int boxStride,
                                       int netW,
                                       int netH,
                                       std::uint8_t* __restrict__ out,
                                       int* __restrict__ counts,
                                       float thr)
{
    const int j = blockIdx.y;
    float* src = up + static_cast<std::size_t>(j) * upStep;
    std::uint8_t* dst = out + static_cast<std::size_t>(j) * netW * netH;

    const float* box = boxes + static_cast<std::size_t>(j) * boxStride;
    const int xa = ceilClamp(static_cast<double>(box[0]), netW);
    const int ya = ceilClamp(static_cast<double>(box[1]), netH);
    const int xb = ceilClamp(static_cast<double>(box[2]), netW);
    const int yb = ceilClamp(static_cast<double>(box[3]), netH);

    const int y = blockIdx.x;
    const bool rowIn = (y >= ya && y < yb) && (xb > xa);
    const std::size_t rowBase = static_cast<std::size_t>(y) * netW;
    int local = 0;

    // Every pixel of the row is written: inside the window it is "logit > thr",
    // outside it is 0 (crop_mask). Doing it as one branch per pixel instead of
    // three separate ranges keeps the "outside" case from ever being skipped --
    // a row can be partly inside the box, and the columns past `xb` still have
    // to be cleared or the raw logit would leak into mask_orig.
    for (int x = static_cast<int>(threadIdx.x); x < netW;
         x += static_cast<int>(blockDim.x))
    {
        const std::size_t i = rowBase + static_cast<std::size_t>(x);
        const bool in = rowIn && x >= xa && x < xb;
        const bool on = in && (src[i] > thr);
        src[i] = on ? 1.0f : 0.0f;
        dst[i] = on ? 255 : 0;
        local += on ? 1 : 0;
    }

    __shared__ int sRed[kBlock1D];
    sRed[threadIdx.x] = local;
    __syncthreads();
    for (int off = static_cast<int>(blockDim.x) >> 1; off > 0; off >>= 1)
    {
        if (static_cast<int>(threadIdx.x) < off)
        {
            sRed[threadIdx.x] += sRed[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && sRed[0] != 0)
    {
        atomicAdd(&counts[j], sRed[0]);
    }
}

__global__ void thresholdMaskKernel(const float* __restrict__ src,
                                    std::uint8_t* __restrict__ dst,
                                    std::size_t elems,
                                    float thr)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elems)
    {
        dst[i] = (src[i] > thr) ? 255 : 0;
    }
}

__global__ void clampKernel(const float* __restrict__ src,
                            std::uint8_t* __restrict__ dst,
                            std::size_t elems)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elems)
    {
        // Same as std::clamp(v, 0.f, 255.f) then a truncating cast.
        float v = src[i];
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        dst[i] = static_cast<std::uint8_t>(v);
    }
}

}  // namespace

cudaError_t binarizeCropMasks(cudaStream_t stream,
                              int n,
                              float* up,
                              std::size_t upStep,
                              const float* boxes,
                              int boxStride,
                              int netW,
                              int netH,
                              std::uint8_t* out,
                              int* counts,
                              float thr)
{
    if (n <= 0)
    {
        return cudaSuccess;
    }
    if (up == nullptr || boxes == nullptr || out == nullptr || counts == nullptr ||
        netW <= 0 || netH <= 0 || boxStride < 4 || upStep < static_cast<std::size_t>(netW) * netH)
    {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(static_cast<unsigned>(netH), static_cast<unsigned>(n));
    binarizeCropMaskKernel<<<grid, kBlock1D, 0, stream>>>(
        up, upStep, boxes, boxStride, netW, netH, out, counts, thr);

    return cudaGetLastError();
}

cudaError_t thresholdMaskToU8(cudaStream_t stream,
                              const float* src,
                              std::uint8_t* dst,
                              std::size_t elems,
                              float thr)
{
    if (elems == 0)
    {
        return cudaSuccess;
    }
    if (src == nullptr || dst == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    std::size_t block = kBlock1D;
    std::size_t grid  = (elems + block - 1) / block;
    if (grid > 65535 * 32)
    {
        // Keep the 1D grid inside the launch limit; the kernel strides nothing,
        // so fall back to a bigger block instead of a strided loop.
        block = 1024;
        grid  = (elems + block - 1) / block;
    }
    thresholdMaskKernel<<<static_cast<unsigned>(grid), static_cast<unsigned>(block),
                          0, stream>>>(src, dst, elems, thr);

    return cudaGetLastError();
}

cudaError_t clampToU8(cudaStream_t stream,
                      const float* src,
                      std::uint8_t* dst,
                      std::size_t elems)
{
    if (elems == 0)
    {
        return cudaSuccess;
    }
    if (src == nullptr || dst == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    const unsigned grid = static_cast<unsigned>((elems + kBlock1D - 1) / kBlock1D);
    clampKernel<<<grid, kBlock1D, 0, stream>>>(src, dst, elems);

    return cudaGetLastError();
}

}  // namespace trt_alpha::kernels::ops
