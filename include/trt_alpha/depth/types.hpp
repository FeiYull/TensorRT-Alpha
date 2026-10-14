// =============================================================================
//  trt_alpha :: depth :: types
// -----------------------------------------------------------------------------
//  DepthMap -- a dense depth map in the SOURCE-IMAGE coordinate system (the
//  inverse letterbox transform is already applied inside the model).
//
//  Layout: data is row-major, exactly height * width floats, unit = metre,
//  value > 0 (the YOLO26 depth head ends with exp(clamp(-4, 5)), so a depth is
//  never zero or negative). No colour / channel description; pure data.
// =============================================================================
#pragma once

#include <cstddef>
#include <vector>

namespace trt_alpha::depth {

//! A dense depth map (source-image pixel coordinates).
struct DepthMap
{
    int width  = 0;                  //!< source-image width
    int height = 0;                  //!< source-image height
    std::vector<float> data;         //!< row-major, height*width, metres

    [[nodiscard]] bool empty() const noexcept
    {
        return width <= 0 || height <= 0 || data.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept { return data.size(); }
};

}  // namespace trt_alpha::depth
