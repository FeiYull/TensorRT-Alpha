// =============================================================================
//  trt_alpha :: obb :: YoloV11Obb（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Obb（见 yolov11_obb.hpp）。
// =============================================================================
#include "yolov11_obb.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolov11_obb", trt_alpha::obb::YoloV11Obb);
