// =============================================================================
//  trt_alpha :: kernels :: cast（实现）
// =============================================================================
#include "trt_alpha/kernels/cast.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <stdexcept>
#include <string>

namespace trt_alpha::kernels {
namespace {

void checkCuda(cudaError_t err, const char* op)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("kernels::") + op + " failed: " +
                                 cudaGetErrorString(err));
    }
}

__global__ void u8ToF32Kernel(const std::uint8_t* __restrict__ src,
                              float* __restrict__ dst,
                              std::size_t count)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
    {
        dst[i] = static_cast<float>(src[i]);
    }
}

}  // namespace

void u8ToF32(cudaStream_t stream,
             const std::uint8_t* src,
             float* dst,
             std::size_t count)
{
    if (count == 0) { return; }
    const int grid = gridSize1D(count, kBlock1D);
    u8ToF32Kernel<<<grid, kBlock1D, 0, stream>>>(src, dst, count);
    checkCuda(cudaGetLastError(), "u8ToF32 launch");
}

}  // namespace trt_alpha::kernels