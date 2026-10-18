// =============================================================================
//  trt_alpha :: cls :: YoloV8Cls（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOV8-cls 的导出口径与 YOLOV26-cls 一致：Resize(短边) + CenterCrop（不是
//  letterbox），output0 = [batch, 1000]，图里已含 Softmax。
//
//  实测（onnxruntime 导出，batch = 2, imgsz = 224）：
//      input  [2, 3, 224, 224]
//      output [2, 1000]
//      与 yolo26n-cls.onnx 逐项一致。
// =============================================================================
#pragma once

#include "yolov26_cls.hpp"

namespace trt_alpha::cls {

//! YOLOV8 分类模型：与 YoloV26Cls 共用同一套预处理与 top-k 后处理。
class YoloV8Cls final : public YoloV26Cls
{
public:
    YoloV8Cls() { m_tag = "yolov8_cls"; }
};

}  // namespace trt_alpha::cls
