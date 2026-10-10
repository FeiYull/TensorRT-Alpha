// =============================================================================
//  trt_alpha :: kernels :: yolo_decode (internal)
// -----------------------------------------------------------------------------
//  Post-processing kernel declarations (used only inside the .cu).
// =============================================================================
#pragma once

#include "trt_alpha/kernels/legacy/postprocess.hpp"

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

//! Decode the YOLOv8-seg head: same as decodeHeadKernel, but it additionally
//! writes numMaskCoeffs mask coefficients at the end of each row. src row
//! layout = [4 + numClasses + numMaskCoeffs].
__global__ void decodeSegHeadKernel(int batchSize, int numClasses, int topK,
                                    float confThresh, const float* __restrict__ src,
                                    int srcRow, int anchors,
                                    int numMaskCoeffs,
                                    float* __restrict__ dst, int dstRow);

__global__ void decodePoseHeadKernel(int batchSize, int topK,
                                     float confThresh, const float* __restrict__ src,
                                     int srcRow, int anchors,
                                     int numKpts,
                                     float* __restrict__ dst, int dstRow);

__global__ void nmsFastKernel(int topK, int batchSize, float iouThresh,
                              float* __restrict__ src, int srcRow);

}  // namespace trt_alpha::kernels::detail
