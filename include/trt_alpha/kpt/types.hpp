// =============================================================================
//  trt_alpha :: kpt :: types
// -----------------------------------------------------------------------------
//  Keypoint —— 一个关键点（原图像素坐标 + 置信度）。
//  KeypointResult —— 一个人的姿态（框 + 关键点）。
//
//  坐标约定：原图像素坐标（letterbox 逆变换已在模型内部完成）。
// =============================================================================
#pragma once

#include "trt_alpha/det/types.hpp"

#include <vector>

namespace trt_alpha::kpt {

//! 一个关键点。
struct Keypoint
{
    float x = 0.f;            //!< 原图像素 x
    float y = 0.f;            //!< 原图像素 y
    float confidence = 0.f;   //!< 置信度
};

//! 一个人的姿态（框 + 关键点）。
struct KeypointResult
{
    det::Detection box;              //!< 人框（复用 Detection，label 通常 0）
    std::vector<Keypoint> keypoints; //!< N 个关键点（COCO: 17）
};

}  // namespace trt_alpha::kpt