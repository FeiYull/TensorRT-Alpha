// =============================================================================
//  trt_alpha :: kernels :: postprocess
// -----------------------------------------------------------------------------
//  后处理算子（CUDA）。所有指针参数都指向【Device 显存】。
//
//  输出布局约定（objects 缓冲区，每张图一段）：
//    [0]                有效框计数 count（decode 用 atomicAdd 累加）
//    [1 + i*objectWidth ..]  第 i 个框：left top right bottom conf label keep
//                            （keep == 0 表示被 NMS 淘汰）
//    count 上限 topK（decode kernel 内已截断）
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace trt_alpha::kernels {

//! YOLOv8 解码头参数。
struct YoloDecodeParams
{
    // default value
    int batch = 1;
    int numClasses = 80;
    int topK = 300;
    float confThreshold = 0.25f;
    float iouThreshold = 0.45f;
};

//! 每个输出行的固定宽度（不含 mask 系数）：
//! left / top / right / bottom / conf / label / keep
constexpr int kObjectWidth = 7;

//! 解码 YOLO-NAS 检测头：
//! 输入 [B, anchors, 4 + numClasses]（3 维，无 objectness，
//! item[0..3] 直接是 left/top/right/bottom，像素坐标）。
//! 输出布局与 decodeYoloV5Head 相同。
void decodeYoloNasHead(cudaStream_t stream, const YoloDecodeParams& p,
                       const float* src, int anchors, float* objects);

//! 解码 YuNet（libfacedetection）检测头。
//! 输入 3 个张量：loc [B, N, 14] / conf [B, N, 2] / iou [B, N, 1]。
//! 输出每行 17 个 float：
//!   [left top right bottom conf label keep] + 5 个关键点 (x, y) 对（10 个）
//! priorBoxes: [N, 4]（Device，Host 算好后上传）
//! variances:  [2]（Device）
void decodeYuNetHead(cudaStream_t stream,
                     const float* loc, const float* conf, const float* iou,
                     int batch, int numCandidates,
                     int srcImgW, int srcImgH,
                     float confThreshold, int topK,
                     const float* priorBoxes,
                     const float* variances,
                     float* objects);

//! 转置 [batch, srcRow, anchors] -> [batch, anchors, srcRow]（Device -> Device）
void transposeAnchors(cudaStream_t stream, int batch,
                      const float* src, int srcRow, int anchors,
                      float* dst);

//! 解码 YOLOv4 检测头：
//! 输入 [B, anchors, 1, 4 + numClasses]（4 维，无 objectness，cx/cy/w/h 归一化 0~1），
//! 输出布局与 decodeYoloV5Head 相同（left top right bottom conf label keep）。
//! cx/cy/w/h 会乘 dstW/dstH 转成像素坐标。
void decodeYoloV4Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors,
                      int dstW, int dstH,
                      float* objects);

//! 解码 anchor-based + objectness 的 YOLO 头（v5 / v7 / v3 / v4 / yolor 通用）：
//! 输入 [B, anchors, 5 + numClasses]（layout 已是 anchor-major，无需 transpose），
//! confidence = objectness × max(cls_score)。
//! 解码 YOLOv5 检测头（anchor-based，含 objectness）：
//! 输入 [B, anchors, 5 + numClasses]（layout 已是 anchor-major，无需 transpose），
//! 输出与 decodeYoloV8Head 相同布局（left top right bottom conf label keep）。
void decodeYoloV5Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors, float* objects);

//! 解码 YOLOv8 检测头（anchor-free，xywh -> xyxy，取类别最大分）
void decodeYoloV8Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors, float* objects);

//! 解码 YOLOv8-seg 头：同上，另把 numMaskCoeffs 个掩码系数写进行尾。
void decodeYoloV8SegHead(cudaStream_t stream, const YoloDecodeParams& p,
                          const float* src, int anchors,
                          int numMaskCoeffs, float* objects);

//! 解码 YOLOv8-pose 头：和 decodeYoloV8Head 一样，但额外把 numKpts 个关键点的
//! (x, y, conf) 写进行尾（网络输入坐标，不做仿射变换）。
//! src 每行布局 = [4 + 1 + numKpts * 3]。
void decodeYoloV8PoseHead(cudaStream_t stream, const YoloDecodeParams& p,
                          const float* src, int anchors,
                          int numKpts, float* objects);

//! u2net 后处理归一化：out = scale * (val - min) / (max - min)
//! 每张图独立（minVals/maxVals 长度 = batch）。
void normPred(cudaStream_t stream, int batch,
              float* data, int width, int height,
              float scale,
              const float* minVals, const float* maxVals);

//! NMS（按类别，O(count^2) kernel）。写 keep=0 淘汰。
//! objectWidth 通常等于 kObjectWidth（seg 版本 = kObjectWidth + numMaskCoeffs）。
void nmsFast(cudaStream_t stream, const YoloDecodeParams& p,
             float* objects, int objectWidth);

}  // namespace trt_alpha::kernels