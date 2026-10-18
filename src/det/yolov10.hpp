// =============================================================================
//  trt_alpha :: det :: YoloV10（私有头文件）
// -----------------------------------------------------------------------------
//  ⚠️ YOLOV10 有两套互不兼容的导出契约，必须确认拿到的是哪一套：
//
//    (A) one2many  [batch, 4 + nc, anchors] = [batch, 84, 8400]   <-- 本实现
//    (B) end2end   [batch, 300, 6]（xyxy, conf, cls，NMS 免费）
//
//  ultralytics 8.4.165 的 `yolo export model=yolov10n.pt format=onnx` 导出的是
//  (A)：`Detect.fuse()` 先把 one2one 提升成 cv2/cv3 并删掉 one2one_cv2，之后
//  `end2end` property 恒为 False，手动 `head.end2end = True` 只会打一条
//  "no one-to-one head" 警告。实测 PyTorch 报 output shape (2, 84, 8400)。
//
//  官方 release 的 yolov10 onnx 是 (B)。若误用 (B)，YoloV26::discoverEngineIo
//  的 expectC = 4 + nc + extra 检查会在启动时直接报错（300 != 84），不会静默
//  算错；要支持 (B) 得另写一条 top-k 解析，不是复用。
// =============================================================================
#pragma once

#include "yolov26.hpp"

namespace trt_alpha::det {

//! YOLOV10 检测模型（one2many 契约）：与 YoloV26 共用同一套前后处理。
class YoloV10 final : public YoloV26
{
public:
    YoloV10() { m_tag = "yolov10"; }
};

}  // namespace trt_alpha::det
