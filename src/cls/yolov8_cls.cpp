// =============================================================================
//  trt_alpha :: cls :: YoloV8Cls（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Cls（见 yolov8_cls.hpp 的说明）。
// =============================================================================
#include "yolov8_cls.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolov8_cls", trt_alpha::cls::YoloV8Cls);
