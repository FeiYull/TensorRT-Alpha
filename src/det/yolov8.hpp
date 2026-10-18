// =============================================================================
//  trt_alpha :: det :: YoloV8（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOV8-det 的 ONNX 契约与 YOLOV26-det 逐项相同：
//      input   [batch, 3, 640, 640]（NCHW，正方形 letterbox，/255）
//      output0 [batch, 4 + nc, anchors] = [batch, 84, 8400]
//  dist2bbox / *stride / sigmoid 已烘焙进图 => box 是 xywh 像素，cls 是概率。
//
//  实测（onnxruntime，同一张 bus.jpg，正方形 letterbox）：
//      top-anchor box = [320.0, 291.4, 469.3, 309.6]（xywh），
//      官方 nms.non_max_suppression 直接可解码，与 yolov26 同一后处理入口。
//
//  所以这里不复制管线，只继承并把标签换成 yolov8 —— 复制一份处理逻辑就等于
//  多一份会漂移的实现。
// =============================================================================
#pragma once

#include "yolov26.hpp"

namespace trt_alpha::det {

//! YOLOV8 检测模型：与 YoloV26 共用同一套前后处理，只有标签不同。
class YoloV8 final : public YoloV26
{
public:
    YoloV8() { m_tag = "yolov8"; }
};

}  // namespace trt_alpha::det
