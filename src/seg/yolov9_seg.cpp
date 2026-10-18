// =============================================================================
//  trt_alpha :: seg :: YoloV9Seg（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Seg（见 yolov9_seg.hpp 的说明）。
// =============================================================================
#include "yolov9_seg.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolov9_seg", trt_alpha::seg::YoloV9Seg);
