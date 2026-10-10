// =============================================================================
//  trt_alpha :: kernels :: u2net_reduce
// -----------------------------------------------------------------------------
//  u2net-specific reduce: per-row max / (min, max).
//  Input [B, N] contiguous, output [B].
//  Replaces thrust::max_element / thrust::minmax_element (thrust does not allow
//  host-side calls to device algorithms under CUDA 12.9 + MSVC).
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace trt_alpha::kernels {

//! Per-row maximum. data [B, N] -> out [B].
void reduceMax(cudaStream_t stream,
               const float* data,
               float* out,
               int batch, int N);

//! Per-row minimum and maximum. data [B, N] -> outMin [B] + outMax [B].
void reduceMinMax(cudaStream_t stream,
                  const float* data,
                  float* outMin,
                  float* outMax,
                  int batch, int N);

}  // namespace trt_alpha::kernels
