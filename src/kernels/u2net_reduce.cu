// =============================================================================
//  trt_alpha :: kernels :: u2net_reduce（实现）
// =============================================================================
#include "trt_alpha/kernels/u2net_reduce.hpp"

#include <cfloat>
#include <stdexcept>
#include <string>

namespace trt_alpha::kernels {
namespace {

constexpr int kBlock = 256;

void checkCuda(cudaError_t err, const char* op)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("kernels::") + op + " failed: " +
                                 cudaGetErrorString(err));
    }
}

//! 每行求 max。一个 block 处理一行。
__global__ void reduceMaxKernel(const float* __restrict__ data,
                                float* __restrict__ out,
                                int N)
{
    const int b = blockIdx.x;
    const float* row = data + static_cast<std::size_t>(b) * N;

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;

    float local = (tid < N) ? row[tid] : -FLT_MAX;
    for (int i = tid + blockDim.x; i < N; i += blockDim.x)
    {
        local = fmaxf(local, row[i]);
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        out[b] = sdata[0];
    }
}

//! 每行求 min + max。一个 block 处理一行。
__global__ void reduceMinMaxKernel(const float* __restrict__ data,
                                   float* __restrict__ outMin,
                                   float* __restrict__ outMax,
                                   int N)
{
    const int b = blockIdx.x;
    const float* row = data + static_cast<std::size_t>(b) * N;

    extern __shared__ float sdata[];   // 2 * blockDim.x
    float* sMin = sdata;
    float* sMax = sdata + blockDim.x;

    const int tid = threadIdx.x;

    float localMin = (tid < N) ? row[tid] : FLT_MAX;
    float localMax = (tid < N) ? row[tid] : -FLT_MAX;
    for (int i = tid + blockDim.x; i < N; i += blockDim.x)
    {
        const float v = row[i];
        localMin = fminf(localMin, v);
        localMax = fmaxf(localMax, v);
    }
    sMin[tid] = localMin;
    sMax[tid] = localMax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sMin[tid] = fminf(sMin[tid], sMin[tid + s]);
            sMax[tid] = fmaxf(sMax[tid], sMax[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        outMin[b] = sMin[0];
        outMax[b] = sMax[0];
    }
}

}  // namespace

void reduceMax(cudaStream_t stream,
               const float* data,
               float* out,
               int batch, int N)
{
    if (batch <= 0 || N <= 0)
    {
        throw std::runtime_error("reduceMax: batch/N must be > 0");
    }
    const std::size_t smem = kBlock * sizeof(float);
    reduceMaxKernel<<<batch, kBlock, smem, stream>>>(data, out, N);
    checkCuda(cudaGetLastError(), "reduceMax launch");
}

void reduceMinMax(cudaStream_t stream,
                  const float* data,
                  float* outMin,
                  float* outMax,
                  int batch, int N)
{
    if (batch <= 0 || N <= 0)
    {
        throw std::runtime_error("reduceMinMax: batch/N must be > 0");
    }
    const std::size_t smem = 2 * kBlock * sizeof(float);
    reduceMinMaxKernel<<<batch, kBlock, smem, stream>>>(data, outMin, outMax, N);
    checkCuda(cudaGetLastError(), "reduceMinMax launch");
}

}  // namespace trt_alpha::kernels