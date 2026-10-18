// =============================================================================
//  trt_alpha :: kpt :: YoloV8Pose（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOV8-pose 的 ONNX 契约与 YOLOV26-pose 相同：
//      output0 [batch, 4 + 1 + 17*3, anchors] = [batch, 56, 8400]
//      关键点三通道是 (x, y, conf)，dist2bbox / kpts_decode 已烘焙进图。
//
//  实测（onnxruntime 导出，batch = 2, imgsz = 640）：
//      [2, 56, 8400]，与 yolo26n-pose.onnx 逐项一致。
//
//  extra 由 num_kpts * 3 推导（17*3 = 51），INI 里不写。
// =============================================================================
#pragma once

#include "yolov26_pose.hpp"

namespace trt_alpha::kpt {

//! YOLOV8 人体关键点模型：与 YoloV26Pose 共用同一套前后处理。
class YoloV8Pose final : public YoloV26Pose
{
public:
    YoloV8Pose() { m_tag = "yolov8_pose"; }
};

}  // namespace trt_alpha::kpt
