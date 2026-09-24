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
    int batch = 1;
    int numClasses = 80;
    int topK = 300;
    float confThreshold = 0.25f;
    float iouThreshold = 0.45f;
};

//! 每个输出行的固定宽度（不含 mask 系数）：
//! left / top / right / bottom / conf / label / keep
constexpr int kObjectWidth = 7;

//! 转置 [batch, srcRow, anchors] -> [batch, anchors, srcRow]（Device -> Device）
void transposeAnchors(cudaStream_t stream, int batch,
                      const float* src, int srcRow, int anchors,
                      float* dst);

//! 解码 YOLOv8 检测头（anchor-free，xywh -> xyxy，取类别最大分）
void decodeYoloV8Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors, float* objects);

//! 解码 YOLOv8-seg 头：同上，另把 numMaskCoeffs 个掩码系数写进行尾。
void decodeYoloV8SegHead(cudaStream_t stream, const YoloDecodeParams& p,
                          const float* src, int anchors,
                          int numMaskCoeffs, float* objects);

//! NMS（按类别，O(count^2) kernel）。写 keep=0 淘汰。
//! objectWidth 通常等于 kObjectWidth（seg 版本 = kObjectWidth + numMaskCoeffs）。
void nmsFast(cudaStream_t stream, const YoloDecodeParams& p,
             float* objects, int objectWidth);

}  // namespace trt_alpha::kernels