// =============================================================================
//  trt_alpha :: kernels :: yolo_preprocess (internal)
// -----------------------------------------------------------------------------
//  Pre-processing kernel declarations (used only inside the .cu, not exported).
// =============================================================================
#pragma once

#include "trt_alpha/kernels/legacy/preprocess.hpp"

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

__global__ void hwcToChwKernel(const float* __restrict__ src,
                               float* __restrict__ dst,
                               int batchSize, int width, int height);

__global__ void divByMaxKernel(int batchSize, float* __restrict__ data,
                               int volume, const float* __restrict__ maxVals);

//! Bilinear letterbox resize kernel.
__global__ void resizeLetterboxKernel(const std::uint8_t* __restrict__ src,
                                      int srcW, int srcH,
                                      float* __restrict__ dst, int dstW, int dstH,
                                      int batchSize, float padValue, AffineMat m);

//! float-input variant (same logic, src is float).
__global__ void resizeLetterboxF32Kernel(const float* __restrict__ src,
                                         int srcW, int srcH,
                                         float* __restrict__ dst, int dstW, int dstH,
                                         int batchSize, float padValue, AffineMat m);

__global__ void resizeNoPaddingRgbKernel(const float* __restrict__ src,
                                         int srcW, int srcH,
                                         float* __restrict__ dst,
                                         int dstW, int dstH,
                                         int batchSize, AffineMat m);

__global__ void resizeNoPaddingGrayKernel(const float* __restrict__ src,
                                          int srcW, int srcH,
                                          float* __restrict__ dst,
                                          int dstW, int dstH,
                                          int batchSize, AffineMat m);

}  // namespace trt_alpha::kernels::detail
