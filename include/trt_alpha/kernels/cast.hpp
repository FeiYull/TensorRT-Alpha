// =============================================================================
//  trt_alpha :: kernels :: cast
// -----------------------------------------------------------------------------
//  Type-conversion kernels (Device -> Device).
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_alpha::kernels {

//! uint8 -> float32 (Device -> Device), element-wise.
void u8ToF32(cudaStream_t stream,
             const std::uint8_t* src,
             float* dst,
             std::size_t count);

}  // namespace trt_alpha::kernels
