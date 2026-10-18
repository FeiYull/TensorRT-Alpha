// =============================================================================
//  trt_alpha :: kpt :: YoloV11Pose（私有头文件）
// -----------------------------------------------------------------------------
//  YOLO11-pose 的 ONNX 契约与 YOLO26-pose 相同：
//      output0 [batch, 4 + 1 + 17*3, anchors] = [batch, 56, 8400]，
//      关键点三通道是 (x, y, conf)，decode 已烘焙进图。
//  实测（onnxruntime 官方 nms + scale_boxes/scale_coords 对拍，8 张混尺寸图）：
//      box <= 3.4e-4 px，kpt <= 3.4e-4 px，conf <= 1.8e-7。
//
//  extra 由 num_kpts * 3 推导（17*3 = 51），INI 里不写。
// =============================================================================
#pragma once

#include "yolov26_pose.hpp"

namespace trt_alpha::kpt {

//! YOLO11 人体关键点模型：与 YoloV26Pose 共用同一套前后处理。
class YoloV11Pose final : public YoloV26Pose
{
public:
    YoloV11Pose() { m_tag = "yolo11_pose"; }
};

}  // namespace trt_alpha::kpt
