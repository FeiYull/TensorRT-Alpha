// =============================================================================
//  trt_alpha :: kpt :: YoloV11Pose（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Pose（见 yolo11_pose.hpp）。
// =============================================================================
#include "yolo11_pose.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolo11_pose", trt_alpha::kpt::YoloV11Pose);
