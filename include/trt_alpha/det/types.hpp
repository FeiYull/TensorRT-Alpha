// =============================================================================
//  trt_alpha :: det :: types
// -----------------------------------------------------------------------------
//  Detection —— 一个检测框（可选带关键点）。
//
//  坐标约定：原图像素坐标（letterbox 逆变换已在模型内部完成）。
//  不再描述颜色 / 通道；纯数据。
// =============================================================================
#pragma once

#include <vector>

namespace trt_alpha::det {

//! 一个 2D 点（原图像素坐标，float）。
struct Point2f
{
    float x = 0.f;
    float y = 0.f;
};

//! 一个检测框。
struct Detection
{
    float left = 0.f;         //!< 左（原图像素 x）
    float top = 0.f;          //!< 上（原图像素 y）
    float right = 0.f;        //!< 右（原图像素 x）
    float bottom = 0.f;       //!< 下（原图像素 y）
    float confidence = 0.f;   //!< 置信度
    int label = -1;           //!< 类别 ID（-1 = 无效）

    //! 可选关键点（人脸 5 点等）。空 = 无关键点（YOLO 系列）。
    std::vector<Point2f> land_marks;
};

}  // namespace trt_alpha::det