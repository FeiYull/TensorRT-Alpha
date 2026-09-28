// =============================================================================
//  trt_alpha :: kernels :: yolo_preprocess（内部）
// -----------------------------------------------------------------------------
//  预处理 kernel 声明（仅 .cu 内部使用，不对外暴露）。
// =============================================================================
#pragma once

#include "trt_alpha/kernels/preprocess.hpp"

#include <cstdint>

namespace trt_alpha::kernels::detail {

__global__ void bgrToRgbHwcKernel(float* __restrict__ data,
                                  int batchSize, int width, int height);

__global__ void copyWithPaddingKernel(int batchSize,
                                      const float* __restrict__ src,
                                      int srcWidth, int srcHeight,
                                      float* __restrict__ dst,
                                      int dstWidth, int dstHeight,
                                      float paddingValue,
                                      int padTop, int padLeft);

__global__ void bgrToNchwNormKernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    int batchSize, int width, int height,
                                    float scale, float m0, float m1, float m2,
                                    float s0, float s1, float s2,
                                    int swapRB);

//! letterbox 双线性 resize kernel。
__global__ void resizeLetterboxKernel(const std::uint8_t* __restrict__ src,
                                      int srcW, int srcH,
                                      float* __restrict__ dst, int dstW, int dstH,
                                      int batchSize, float padValue, AffineMat m);

//! float 输入版（逻辑同上，只是 src 是 float）。
__global__ void resizeLetterboxF32Kernel(const float* __restrict__ src,
                                         int srcW, int srcH,
                                         float* __restrict__ dst, int dstW, int dstH,
                                         int batchSize, float padValue, AffineMat m);

}  // namespace trt_alpha::kernels::detail