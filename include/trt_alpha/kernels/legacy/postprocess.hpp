// =============================================================================
//  trt_alpha :: kernels :: postprocess
// -----------------------------------------------------------------------------
//  Postprocessing operators (CUDA). Every pointer argument refers to
//  [Device memory].
//
//  Output layout convention (objects buffer, one segment per image):
//    [0]                      valid box count (accumulated with atomicAdd during decode)
//    [1 + i*objectWidth ..]   the i-th box: left top right bottom conf label keep
//                             (keep == 0 means suppressed by NMS)
//    count is capped at topK (truncated inside the decode kernel)
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace trt_alpha::kernels {

//! Parameters for the YOLOv8 decode head.
struct YoloDecodeParams
{
    // default value
    int batch = 1;
    int numClasses = 80;
    int topK = 300;
    float confThreshold = 0.25f;
    float iouThreshold = 0.45f;
};

//! Fixed width of each output row (excluding mask coefficients):
//! left / top / right / bottom / conf / label / keep
constexpr int kObjectWidth = 7;

//! Decode the YOLO-NAS detection head:
//! input [B, anchors, 4 + numClasses] (3-D, no objectness;
//! item[0..3] are left/top/right/bottom directly, in pixel coords).
//! Output layout matches decodeYoloV5Head.
void decodeYoloNasHead(cudaStream_t stream, const YoloDecodeParams& p,
                       const float* src, int anchors, float* objects);

//! Fixed row widths for YuNet's three outputs (shared between the decode
//! kernel and the engine-shape validation in yunet.cpp).
//! WARNING: this is a hard assumption of the kernel: when it disagrees with
//! the engine declaration, the kernel reads/writes out of bounds.
//! yunet.cpp therefore validates it against the engine shape after
//! setInputShape -- single source of truth, never write a second copy.
constexpr int kYuNetLocRow     = 14;   //!< 4 bbox + 10 landmark
constexpr int kYuNetConfRow    = 2;    //!< background / face
constexpr int kYuNetIouRow     = 1;    //!< IoU branch
constexpr int kYuNetObjectsRow = 17;   //!< 4 bbox + conf + label + keep + 10 landmark

//! Decode the YuNet (libfacedetection) detection head.
//! Inputs: 3 tensors loc [B, N, 14] / conf [B, N, 2] / iou [B, N, 1].
//! Each output row holds 17 floats:
//!   [left top right bottom conf label keep] + 5 keypoints (x, y) pairs (10 values)
//! priorBoxes: [N, 4] (Device; computed on Host and uploaded)
//! variances:  [2] (Device)
void decodeYuNetHead(cudaStream_t stream,
                     const float* loc, const float* conf, const float* iou,
                     int batch, int numCandidates,
                     int srcImgW, int srcImgH,
                     float confThreshold, int topK,
                     const float* priorBoxes,
                     const float* variances,
                     float* objects);

//! Transpose [batch, srcRow, anchors] -> [batch, anchors, srcRow] (Device -> Device)
void transposeAnchors(cudaStream_t stream, int batch,
                      const float* src, int srcRow, int anchors,
                      float* dst);

//! Decode the YOLOv4 detection head:
//! input [B, anchors, 1, 4 + numClasses] (4-D, no objectness, cx/cy/w/h normalized 0~1),
//! output layout matches decodeYoloV5Head (left top right bottom conf label keep).
//! cx/cy/w/h are multiplied by dstW/dstH to convert to pixel coords.
void decodeYoloV4Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors,
                      int dstW, int dstH,
                      float* objects);

//! Decode an anchor-based + objectness YOLO head (shared by v5 / v7 / v3 / v4 / yolor):
//! input [B, anchors, 5 + numClasses] (layout is already anchor-major, no transpose),
//! confidence = objectness x max(cls_score).
//! Decode the YOLOv5 detection head (anchor-based, with objectness):
//! input [B, anchors, 5 + numClasses] (layout is already anchor-major, no transpose),
//! output layout matches decodeYoloV8Head (left top right bottom conf label keep).
void decodeYoloV5Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors, float* objects);

//! Decode the YOLOv8 detection head (anchor-free, xywh -> xyxy, best class score)
void decodeYoloV8Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors, float* objects);

//! Decode the YOLOv8-seg head: same as above, plus numMaskCoeffs mask coefficients
//! appended to the row tail.
void decodeYoloV8SegHead(cudaStream_t stream, const YoloDecodeParams& p,
                          const float* src, int anchors,
                          int numMaskCoeffs, float* objects);

//! Decode the YOLOv8-pose head: same as decodeYoloV8Head, but additionally appends
//! the (x, y, conf) of numKpts keypoints to the row tail (network input coords,
//! no affine transform).
//! Per-row layout of src = [4 + 1 + numKpts * 3].
void decodeYoloV8PoseHead(cudaStream_t stream, const YoloDecodeParams& p,
                          const float* src, int anchors,
                          int numKpts, float* objects);

//! u2net postprocess normalization: out = scale * (val - min) / (max - min)
//! Independent per image (minVals/maxVals length = batch).
void normPred(cudaStream_t stream, int batch,
              float* data, int width, int height,
              float scale,
              const float* minVals, const float* maxVals);

//! NMS (per-class, O(count^2) kernel). Sets keep=0 to suppress.
//! objectWidth is usually kObjectWidth (seg variant = kObjectWidth + numMaskCoeffs).
void nmsFast(cudaStream_t stream, const YoloDecodeParams& p,
             float* objects, int objectWidth);

}  // namespace trt_alpha::kernels
