// =============================================================================
//  trt_alpha :: det :: types
// -----------------------------------------------------------------------------
//  Detection -- one detection box (optionally carrying keypoints).
//
//  Coordinate convention: source-image pixel coordinates (the inverse
//  letterbox transform is already applied inside the model).
//  No colour / channel description; pure data.
// =============================================================================
#pragma once

#include <vector>

namespace trt_alpha::det {

//! A 2D point (source-image pixel coordinates, float).
struct Point2f
{
    float x = 0.f;
    float y = 0.f;
};

//! One detection box.
struct Detection
{
    float left = 0.f;         //!< left (source-image pixel x)
    float top = 0.f;          //!< top (source-image pixel y)
    float right = 0.f;        //!< right (source-image pixel x)
    float bottom = 0.f;       //!< bottom (source-image pixel y)
    float confidence = 0.f;   //!< confidence
    int label = -1;           //!< class ID (-1 = invalid)

    //! Optional keypoints (e.g. the 5 face landmarks). Empty = no keypoints
    //! (YOLO series).
    std::vector<Point2f> land_marks;
};

}  // namespace trt_alpha::det
