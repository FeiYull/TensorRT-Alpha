// =============================================================================
//  bgr_to_rgba（实现）
// -----------------------------------------------------------------------------
//  One thread per output pixel, purely memory bound: 6.2 MB read + 8.3 MB write
//  at 1080p, ~0.03 ms on a 5060 Ti. The write goes into the mapped PBO (a plain
//  device pointer), so no surface object and no read-only cudaArray to fight --
//  this is the classic, well-trodden CUDA-GL interop path. glTexSubImage2D() then
//  DMAs the PBO into the texture on the GPU, no D2H anywhere.
//
//  Channel order: we store the BGR input in the R/G/B channels unchanged and set
//  alpha to 255. The video fragment shader applies `.bgr`, which recovers true
//  RGB. See bgr_to_rgba.hpp for why the two upload paths (host / device) must
//  agree on this encoding.
// =============================================================================
#include "bgr_to_rgba.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::renderer {
namespace {

__global__ void bgrToRgbaKernel(const unsigned char* __restrict__ src,
                                uchar4* __restrict__ dst,
                                int srcStride,
                                int width,
                                int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    const unsigned char* row =
        src + static_cast<std::size_t>(y) * static_cast<std::size_t>(srcStride) +
        static_cast<std::size_t>(x) * 3u;
    dst[static_cast<std::size_t>(y) * width + x] =
        make_uchar4(row[0], row[1], row[2], 255u);
}

}  // namespace

cudaError_t bgr8ToRgba(cudaStream_t stream,
                       const std::uint8_t* src,
                       int srcStride,
                       std::uint8_t* dst,
                       int width,
                       int height)
{
    if (src == nullptr || dst == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (width <= 0 || height <= 0 || srcStride < width * 3)
    {
        return cudaErrorInvalidValue;
    }

    const dim3 block(32u, 8u);
    const dim3 grid(static_cast<unsigned>((width + 31) / 32),
                    static_cast<unsigned>((height + 7) / 8));

    bgrToRgbaKernel<<<grid, block, 0, stream>>>(
        src, reinterpret_cast<uchar4*>(dst), srcStride, width, height);

    return cudaGetLastError();
}

}  // namespace trt_alpha::renderer
