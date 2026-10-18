// =============================================================================
//  trt_alpha :: cls :: YoloV11Cls（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Cls（见 yolov11_cls.hpp）。
// =============================================================================
#include "yolov11_cls.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolov11_cls", trt_alpha::cls::YoloV11Cls);
