// =============================================================================
//  trt_alpha :: obb :: YoloV11Obb（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Obb（见 yolo11_obb.hpp）。
// =============================================================================
#include "yolo11_obb.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolo11_obb", trt_alpha::obb::YoloV11Obb);
