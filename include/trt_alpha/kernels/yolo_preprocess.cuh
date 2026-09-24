// =============================================================================
//  trt_alpha :: kernels :: yolo_preprocess（内部）
// -----------------------------------------------------------------------------
//  预处理 kernel 声明（仅 .cu 内部使用，不对外暴露）。
// =============================================================================
#pragma once

#include "trt_alpha/kernels/preprocess.hpp"

#include <cstdint>

namespace trt_alpha::kernels::detail {

//! letterbox 双线性 resize kernel。
__global__ void resizeLetterboxKernel(const std::uint8_t* __restrict__ src,
                                      int srcW, int srcH,
                                      float* __restrict__ dst, int dstW, int dstH,
                                      int batchSize, float padValue, AffineMat m);

//! 融合 kernel：BGR HWC -> 归一化 + BGR→RGB + NCHW。
__global__ void bgrToNchwNormKernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    int batchSize, int width, int height,
                                    float scale, float m0, float m1, float m2,
                                    float s0, float s1, float s2);

}  // namespace trt_alpha::kernels::detail