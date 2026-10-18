// =============================================================================
//  trt_alpha :: det :: YoloV9（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOV9-det 的 ONNX 契约与 YOLOV26-det 逐项相同：
//      input   [batch, 3, 640, 640]（NCHW，正方形 letterbox，/255）
//      output0 [batch, 4 + nc, anchors] = [batch, 84, 8400]
//  DFL / dist2bbox / *stride / sigmoid 已烘焙进图 => box 是 xywh 像素。
//
//  实测（ultralytics 8.4.165 导出 yolov9t.pt，batch = 2, imgsz = 640）：
//      PyTorch output shape (2, 84, 8400)，与 yolo26n.onnx 逐项一致。
//  yolov9 的 GELAN / Silence 只影响主干与层名，不影响输出契约。
// =============================================================================
#pragma once

#include "yolov26.hpp"

namespace trt_alpha::det {

//! YOLOV9 检测模型：与 YoloV26 共用同一套前后处理，只有标签不同。
class YoloV9 final : public YoloV26
{
public:
    YoloV9() { m_tag = "yolov9"; }
};

}  // namespace trt_alpha::det
