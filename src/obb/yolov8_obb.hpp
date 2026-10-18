// =============================================================================
//  trt_alpha :: obb :: YoloV8Obb（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOV8-obb（DOTA v1，nc = 15）的 ONNX 契约与 YOLOV26-obb 相同：
//      output0 [batch, 4 + nc + 1, anchors] = [batch, 20, 21504]（imgsz = 1024）
//      最后一个通道是弧度角，angle = (sigmoid - 0.25) * pi，dist2rbox 已烘焙进图。
//
//  实测（onnxruntime 导出，batch = 2, imgsz = 1024）：
//      [2, 20, 21504]，与 yolo26n-obb.onnx 逐项一致。
//
//  ⚠️ 输入分辨率是 1024（DOTA 的训练口径），不是 640。
//  ⚠️ NMS 走 Fast-NMS(probiou)，不是贪心；由 YoloV26Obb 保证，子类不要碰。
// =============================================================================
#pragma once

#include "yolov26_obb.hpp"

namespace trt_alpha::obb {

//! YOLOV8 旋转框模型：与 YoloV26Obb 共用同一套前后处理。
class YoloV8Obb final : public YoloV26Obb
{
public:
    YoloV8Obb() { m_tag = "yolov8_obb"; }
};

}  // namespace trt_alpha::obb
