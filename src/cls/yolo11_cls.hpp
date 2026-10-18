// =============================================================================
//  trt_alpha :: cls :: YoloV11Cls（私有头文件）
// -----------------------------------------------------------------------------
//  YOLO11-cls 的导出口径与 YOLO26-cls 一致：Resize(短边) + CenterCrop（不是
//  letterbox），output0 = [batch, 1000]，图里已含 Softmax（实测 sum == 1）。
//  与 onnxruntime 同图对拍：argmax 相同，概率差在 1e-3 量级（OpenCV/PIL 的
//  resize 差异 + TRT 自身）。
//
//  所以只继承、不复制管线；真分叉了再 override。
// =============================================================================
#pragma once

#include "yolov26_cls.hpp"

namespace trt_alpha::cls {

//! YOLO11 分类模型：与 YoloV26Cls 共用同一套预处理与 top-k 后处理。
class YoloV11Cls final : public YoloV26Cls
{
public:
    YoloV11Cls() { m_tag = "yolo11_cls"; }
};

}  // namespace trt_alpha::cls
