// =============================================================================
//  trt_alpha :: seg :: YoloV9Seg（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOV9-seg 的 ONNX 契约与 YOLOV26-seg 逐项相同：
//      output0 [batch, 4 + nc + 32, anchors] = [batch, 116, 8400]
//      output1 [batch, 32, 160, 160]          （mask prototype）
//
//  实测（ultralytics 8.4.165 导出 yolov9c-seg.pt，batch = 2, imgsz = 640）：
//      输入 [2,3,640,640]，输出 [2,116,8400] + [2,32,160,160]，
//      与 yolo26n-seg.onnx / yolov8n-seg.onnx 逐项一致。
//
//  yolov9 的 head 仍是 Segment（与 yolov8 同一个类），GELAN 主干只改变层名。
// =============================================================================
#pragma once

#include "yolov26_seg.hpp"

namespace trt_alpha::seg {

//! YOLOV9 实例分割模型：与 YoloV26Seg 共用同一套前后处理。
class YoloV9Seg final : public YoloV26Seg
{
public:
    YoloV9Seg() { m_tag = "yolov9_seg"; }
};

}  // namespace trt_alpha::seg
