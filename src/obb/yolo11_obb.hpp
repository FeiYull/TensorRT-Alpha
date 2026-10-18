// =============================================================================
//  trt_alpha :: obb :: YoloV11Obb（私有头文件）
// -----------------------------------------------------------------------------
//  YOLO11-obb（DOTA v1，nc = 15）的 ONNX 契约与 YOLO26-obb 相同：
//      output0 [batch, 4 + nc + 1, anchors] = [batch, 20, 21504]（imgsz = 1024），
//      最后一个通道是弧度角，dist2rbox 已烘焙进图。
//  实测（onnxruntime 官方 nms(rotated=True) + scale_boxes(xywh=True) 对拍）：
//      centre <= 3.1e-5 px，size <= 3.1e-5 px，angle <= 2.7e-7 rad。
//
//  ⚠️ NMS 是 Fast-NMS(probiou)，不是贪心；这条由 YoloV26Obb 的后处理保证，
//  子类不需要也不应该碰。
// =============================================================================
#pragma once

#include "yolov26_obb.hpp"

namespace trt_alpha::obb {

//! YOLO11 旋转框模型：与 YoloV26Obb 共用同一套前后处理。
class YoloV11Obb final : public YoloV26Obb
{
public:
    YoloV11Obb() { m_tag = "yolo11_obb"; }
};

}  // namespace trt_alpha::obb
