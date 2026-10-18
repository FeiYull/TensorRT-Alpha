// =============================================================================
//  trt_alpha :: obb :: YoloV8Obb（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Obb（见 yolov8_obb.hpp 的说明）。
// =============================================================================
#include "yolov8_obb.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolov8_obb", trt_alpha::obb::YoloV8Obb);
