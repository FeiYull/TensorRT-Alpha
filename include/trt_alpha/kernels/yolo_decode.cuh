// =============================================================================
//  trt_alpha :: kernels :: yolo_decode（内部）
// -----------------------------------------------------------------------------
//  后处理 kernel 声明（仅 .cu 内部使用）。
// =============================================================================
#pragma once

#include "trt_alpha/kernels/postprocess.hpp"

namespace trt_alpha::kernels::detail {

__global__ void decodeNasHeadKernel(int batchSize, int numClasses, int topK,
                                    float confThresh, const float* __restrict__ src,
                                    int srcRow, int anchors,
                                    float* __restrict__ dst, int dstRow);

__global__ void decodeYuNetKernel(
    int batchSize, int numCandidates, int topK, float confThreshold,
    int srcImgW, int srcImgH,
    const float* __restrict__ loc,  int locRow,
    const float* __restrict__ conf, int confRow,
    const float* __restrict__ iou,  int iouRow,
    const float* __restrict__ priorBoxes,
    const float* __restrict__ variances,
    float* __restrict__ dst, int dstRow);

__global__ void normPredKernel(int batchSize, float* __restrict__ data,
                               int area, float scale,
                               const float* __restrict__ minVals,
                               const float* __restrict__ maxVals);

__global__ void transposeKernel(int batchSize, const float* __restrict__ src,
                                int srcRow, int anchors, float* __restrict__ dst);

__global__ void decodeV4HeadKernel(int batchSize, int numClasses, int topK,
                                   float confThresh, const float* __restrict__ src,
                                   int anchors, int dstW, int dstH,
                                   float* __restrict__ dst, int dstRow);

__global__ void decodeV5HeadKernel(int batchSize, int numClasses, int topK,
                                float confThresh, const float* __restrict__ src,
                                int srcRow, int anchors,
                                float* __restrict__ dst, int dstRow);

__global__ void decodeHeadKernel(int batchSize, int numClasses, int topK,
                                 float confThresh, const float* __restrict__ src,
                                 int srcRow, int anchors,
                                 float* __restrict__ dst, int dstRow,
                                 int numMaskCoeffs);

//! 解码 YOLOv8-seg 头：和 decodeHeadKernel 一样，但额外把 numMaskCoeffs 个 mask 系数
//! 写进行尾。src 的每行布局 = [4 + numClasses + numMaskCoeffs]。
__global__ void decodeSegHeadKernel(int batchSize, int numClasses, int topK,
                                    float confThresh, const float* __restrict__ src,
                                    int srcRow, int anchors,
                                    int numMaskCoeffs,
                                    float* __restrict__ dst, int dstRow);

__global__ void nmsFastKernel(int topK, int batchSize, float iouThresh,
                              float* __restrict__ src, int srcRow);

}  // namespace trt_alpha::kernels::detail