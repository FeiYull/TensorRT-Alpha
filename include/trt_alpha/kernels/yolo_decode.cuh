// =============================================================================
//  trt_alpha :: kernels :: yolo_decode（内部）
// -----------------------------------------------------------------------------
//  后处理 kernel 声明（仅 .cu 内部使用）。
// =============================================================================
#pragma once

#include "trt_alpha/kernels/postprocess.hpp"

namespace trt_alpha::kernels::detail {

__global__ void transposeKernel(int batchSize, const float* __restrict__ src,
                                int srcRow, int anchors, float* __restrict__ dst);

__global__ void decodeHeadKernel(int batchSize, int numClasses, int topK,
                                 float confThresh, const float* __restrict__ src,
                                 int srcRow, int anchors,
                                 float* __restrict__ dst, int dstRow,
                                 int numMaskCoeffs);

__global__ void nmsFastKernel(int topK, int batchSize, float iouThresh,
                              float* __restrict__ src, int srcRow);

}  // namespace trt_alpha::kernels::detail