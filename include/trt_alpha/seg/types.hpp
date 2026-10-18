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
    det::Detection box;                        //!< box, in SOURCE-IMAGE pixels (label == -1 means no box)

    //! Binary mask in NETWORK-INPUT space (e.g. 640x640, letterbox padding included),
    //! CV_8UC1 semantics: 255 = foreground, 0 = background. This is the raw layout
    //! that matches the python reference / test dumps -- keep using it for parity.
    core::BufferView mask;
    std::shared_ptr<core::Buffer> maskOwner;   //!< owner of `mask`

    //! The same mask already un-letterboxed to the SOURCE-IMAGE size (srcW x srcH),
    //! CV_8UC1 0/255 -- usable directly (overlay / crop by `box` / cv::imwrite)
    //! without any resizing on the caller's side. `box` above lives in the very same
    //! source-image coordinate system. Empty for models without an instance mask,
    //! and EMPTY BY DEFAULT for yolo26-seg / yolo11-seg: producing it costs a full
    //! resolution resize per instance, so it is opt-in via `[model] mask_orig = true`.
    core::BufferView maskOrig;
    std::shared_ptr<core::Buffer> maskOrigOwner;   //!< owner of `maskOrig`
};

}  // namespace trt_alpha::seg
