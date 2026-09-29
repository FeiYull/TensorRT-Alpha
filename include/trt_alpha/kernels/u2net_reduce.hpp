// =============================================================================
//  trt_alpha :: kernels :: u2net_reduce
// -----------------------------------------------------------------------------
//  u2net 专用 reduce：每行求 max / (min, max)。
//  输入 [B, N] 连续，输出 [B]。
//  替代 thrust::max_element / thrust::minmax_element
//  （CUDA 12.9 + MSVC 下 thrust 主机端调 device 算法不支持）。
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace trt_alpha::kernels {

//! 每行求最大值。data [B, N] -> out [B]。
void reduceMax(cudaStream_t stream,
               const float* data,
               float* out,
               int batch, int N);

//! 每行求最小值 + 最大值。data [B, N] -> outMin [B] + outMax [B]。
void reduceMinMax(cudaStream_t stream,
                  const float* data,
                  float* outMin,
                  float* outMax,
                  int batch, int N);

}  // namespace trt_alpha::kernels