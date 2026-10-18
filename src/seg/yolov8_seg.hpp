// =============================================================================
//  trt_alpha :: seg :: YoloV8Seg（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOV8-seg 的 ONNX 契约与 YOLOV26-seg 逐项相同：
//      output0 [batch, 4 + nc + 32, anchors] = [batch, 116, 8400]
//      output1 [batch, 32, 160, 160]          （mask prototype，1/4 分辨率）
//  box 是 xywh 像素、cls 是概率、mask 系数在通道尾部 [4+nc, 4+nc+32)。
//
//  实测（onnxruntime 导出 + I/O 对比，batch = 2, imgsz = 640）：
//      与 yolo26n-seg.onnx 的输入/输出形状、通道含义逐项一致。
// =============================================================================
#pragma once

#include "yolov26_seg.hpp"

namespace trt_alpha::seg {

//! YOLOV8 实例分割模型：与 YoloV26Seg 共用同一套前后处理。
class YoloV8Seg final : public YoloV26Seg
{
public:
    YoloV8Seg() { m_tag = "yolov8_seg"; }
};

}  // namespace trt_alpha::seg
