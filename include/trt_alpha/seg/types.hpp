// =============================================================================
//  trt_alpha :: seg :: types
// -----------------------------------------------------------------------------
//  Segmentation -- one segmentation result.
//
//  Fields:
//    * box       -- detection box (optional). label == -1 means "no box"
//                   (e.g. U2Net saliency detection).
//    * mask      -- mask view (CV_8UC1 semantics, full image).
//                   - YOLOv8-seg: binary 0/255 (255 = foreground, 0 = background),
//                     one image per instance
//                   - U2Net:      0~255 saliency
//    * maskOwner -- owner of the mask; keeps mask.data valid.
//
//  Renderer convention:
//    * draw a box when box.label >= 0; otherwise only overlay the mask.
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/det/types.hpp"

#include <memory>

namespace trt_alpha::seg {

//! One segmentation result.
struct Segmentation
{
    det::Detection box;                        //!< box (label == -1 means no box)
    core::BufferView mask;                     //!< mask view (CV_8UC1 semantics, full image)
    std::shared_ptr<core::Buffer> maskOwner;   //!< owner of the mask
};

}  // namespace trt_alpha::seg
