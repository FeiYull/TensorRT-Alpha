// =============================================================================
//  trt_alpha :: kpt :: types
// -----------------------------------------------------------------------------
//  Keypoint       -- one keypoint (source-image pixel coordinates + confidence).
//  KeypointResult -- one person's pose (box + keypoints).
//
//  Coordinate convention: source-image pixel coordinates (the inverse letterbox
//  transform is already applied inside the model).
// =============================================================================
#pragma once

#include "trt_alpha/det/types.hpp"

#include <vector>

namespace trt_alpha::kpt {

//! One keypoint.
struct Keypoint
{
    float x = 0.f;            //!< source-image pixel x
    float y = 0.f;            //!< source-image pixel y
    float confidence = 0.f;   //!< confidence
};

//! One person's pose (box + keypoints).
struct KeypointResult
{
    det::Detection box;              //!< person box (reuses Detection; label is usually 0)
    std::vector<Keypoint> keypoints; //!< N keypoints (COCO: 17)
};

}  // namespace trt_alpha::kpt
