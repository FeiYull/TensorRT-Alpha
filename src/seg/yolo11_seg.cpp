// =============================================================================
//  trt_alpha :: seg :: YoloV11Seg（注册）
// -----------------------------------------------------------------------------
//  只做注册：实现全部继承自 YoloV26Seg（见 yolo11_seg.hpp）。
// =============================================================================
#include "yolo11_seg.hpp"
#include "trt_alpha/core/model_registry.hpp"

TRT_ALPHA_REGISTER_MODEL("yolo11_seg", trt_alpha::seg::YoloV11Seg);
