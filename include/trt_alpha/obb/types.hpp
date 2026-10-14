// =============================================================================
//  trt_alpha :: obb :: types
// -----------------------------------------------------------------------------
//  ObbDetection -- one oriented (rotated) bounding box.
//
//  Coordinate convention: source-image pixel coordinates (the inverse
//  letterbox transform is already applied inside the model).
//  Format: cx, cy, w, h, angle (radians), the same as ultralytics' xywhr.
//  angle is normalised to [0, pi/2) by regularize_rboxes (w/h swapped where
//  needed), so consumers may treat w as the long-ish side convention freely.
// =============================================================================
#pragma once

namespace trt_alpha::obb {

//! One oriented bounding box.
struct ObbDetection
{
    float cx = 0.f;          //!< centre x (source-image pixel)
    float cy = 0.f;          //!< centre y (source-image pixel)
    float w  = 0.f;          //!< width  (pixels)
    float h  = 0.f;          //!< height (pixels)
    float angle = 0.f;       //!< rotation, radians, in [0, pi/2)
    float confidence = 0.f;  //!< confidence
    int   label = -1;        //!< class ID (-1 = invalid)
};

}  // namespace trt_alpha::obb
