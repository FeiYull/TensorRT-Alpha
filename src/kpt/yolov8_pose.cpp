// =============================================================================
//  trt_alpha :: kpt :: YoloV8Pose（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Pose（见 yolov8_pose.hpp 的说明）。
// =============================================================================
#include "yolov8_pose.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolov8_pose", trt_alpha::kpt::YoloV8Pose);
