// =============================================================================
//  trt_alpha :: seg :: YoloV11Seg（私有头文件）
// -----------------------------------------------------------------------------
//  YOLO11-seg 的 ONNX 契约与 YOLO26-seg 逐项相同：
//      output0 [batch, 4 + nc + 32, anchors]，output1 [batch, 32, 160, 160]。
//  实测（onnxruntime 官方 process_mask 对拍，8 张混尺寸图）：
//      mask 0 / 2048000 像素不一致，box <= 2.4e-4 px，conf <= 1.2e-6。
//
//  ⚠️ 掩码顺序是「先 bilinear 上采样到 640，再按框裁剪」，与上游 #24272 之后的
//  实现一致；仓库里的 yolov8_seg 还是老顺序（先裁后缩），别混用。
// =============================================================================
#pragma once

#include "yolov26_seg.hpp"

namespace trt_alpha::seg {

//! YOLO11 实例分割模型：与 YoloV26Seg 共用同一套前后处理。
class YoloV11Seg final : public YoloV26Seg
{
public:
    YoloV11Seg() { m_tag = "yolo11_seg"; }
};

}  // namespace trt_alpha::seg
