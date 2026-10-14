// =============================================================================
//  trt_alpha :: sem :: types
// -----------------------------------------------------------------------------
//  SemanticMask -- a dense semantic segmentation map in the SOURCE-IMAGE
//  coordinate system (the inverse letterbox transform is already applied
//  inside the model).
//
//  Layout: data is row-major, exactly height * width bytes, one class id per
//  pixel (the YOLO26 sem engine emits an already-argmax'ed uint8 map, so no
//  further decoding is needed). Class ids fit in a byte (nc <= 255).
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_alpha::sem {

//! A dense semantic segmentation map (source-image pixel coordinates).
struct SemanticMask
{
    int width  = 0;                        //!< source-image width
    int height = 0;                        //!< source-image height
    std::vector<std::uint8_t> data;        //!< row-major, height*width class ids

    [[nodiscard]] bool empty() const noexcept
    {
        return width <= 0 || height <= 0 || data.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept { return data.size(); }
};

}  // namespace trt_alpha::sem
