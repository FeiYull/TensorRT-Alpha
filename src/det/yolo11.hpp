// =============================================================================
//  trt_alpha :: det :: YoloV11（私有头文件）
// -----------------------------------------------------------------------------
//  YOLO11 与 YOLO26 的导出口径完全一致：
//      * ONNX 已经把 decode 烘焙进图，output0 = [batch, 4 + nc, anchors]；
//      * 静态引擎只接受固定尺寸，两边都走正方形 letterbox；
//      * 后处理同样是 conf 过滤 + 贪心 NMS + scale_boxes。
//
//  实测（onnxruntime + 官方 ops/nms 对拍，8 张混尺寸图，--noTF32 引擎）：
//      seg  mask 0/2048000 像素不一致，box <= 2.4e-4 px；
//      pose box <= 3.4e-4 px，kpt <= 3.4e-4 px；
//      obb  centre <= 3.1e-5 px，angle <= 2.7e-7 rad。
//  所以这里不复制管线，只继承并把标签换成 yolo11 —— 复制一份处理逻辑就等于
//  多一份需要同步维护的真相。真到某天输出布局变了，再在子类里显式 override
//  对应阶段，而不是现在就预防性地抄一遍。
// =============================================================================
#pragma once

#include "yolov26.hpp"

namespace trt_alpha::det {

//! YOLO11 检测模型：与 YoloV26 共用同一套前后处理，只有标签不同。
class YoloV11 final : public YoloV26
{
public:
    YoloV11() { m_tag = "yolo11"; }
};

}  // namespace trt_alpha::det
