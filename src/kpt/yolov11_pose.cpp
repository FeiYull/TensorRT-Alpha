// =============================================================================
//  trt_alpha :: kpt :: YoloV11Pose（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Pose（见 yolov11_pose.hpp）。
// =============================================================================
#include "yolov11_pose.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolov11_pose", trt_alpha::kpt::YoloV11Pose);
