// =============================================================================
//  trt_alpha :: kernels :: ops :: overlay
// -----------------------------------------------------------------------------
//  Baking the task overlays (instance masks / depth heat / semantic colours)
//  into the BGR frame, on the DEVICE. This is the single implementation both
//  renderers use; the old host-side painting in `opencv_renderer.cpp` is kept in
//  the tree for reference but is no longer called.
//
//  Everything here works on a BGR8 frame (`bgr`, `stride` bytes per row). The
//  arithmetic mirrors the python reference (ultralytics Annotator) so the result
//  matches what the host code produced, pixel for pixel:
//      mcs  = amax_i(mask_i * colour_i)
//      inv  = prod_i(1 - mask_i * alpha)
//      out  = (im / 255 * inv + mcs) * 255          (truncating cast)
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace trt_alpha::kernels::ops {

//! One instance mask, already in the layout the postprocess leaves it in:
//! NETWORK space (letterbox padding included), 0 / 255, `stride` bytes per row.
struct SegLayer
{
    const std::uint8_t* mask = nullptr;  //!< device pointer
    int         maskW    = 0;
    int         maskH    = 0;
    std::size_t maskStride = 0;

    //! Content window of the letterbox inside the mask (== computeLetterBox()).
    int newW = 0;
    int newH = 0;
    int padL = 0;
    int padT = 0;

    //! Box in SOURCE pixels, half open [x0, x1) x [y0, y1). Pixels outside it are
    //! skipped outright -- the postprocess already cropped the mask to this box.
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;

    //! colour * alpha, BGR order (the host path stored the same premultiplied
    //! values in a cv::Vec3f before compositing).
    float c0 = 0.f;
    float c1 = 0.f;
    float c2 = 0.f;
};

// =============================================================================
//! Bake `n` instance masks into the frame.
// -----------------------------------------------------------------------------
//! `layers` must point at a device array of SegLayer. `alpha` is the mask
//! opacity (the `mask_alpha` config, 0.5 by default) and is only used to derive
//! `inv = prod(1 - alpha)`; the per-layer colours already carry alpha.
//! Returns cudaSuccess on success; never synchronises.
// =============================================================================
cudaError_t composeSegMasks(cudaStream_t stream,
                            std::uint8_t* bgr,
                            int width,
                            int height,
                            std::size_t stride,
                            const SegLayer* layers,
                            int n,
                            float alpha);

// =============================================================================
//! Bake a semantic colour map into the frame.
// -----------------------------------------------------------------------------
//! `sem` is a device array of `elems` class ids at frame resolution. Id 255 is
//! the ignore index and paints black; everything else picks the ultralytics
//! 20-colour palette. The blend is `addWeighted(im, 1-alpha, colour, alpha, 0)`,
//! i.e. rounded per channel.
// =============================================================================
cudaError_t composeSemantics(cudaStream_t stream,
                             std::uint8_t* bgr,
                             int width,
                             int height,
                             std::size_t stride,
                             const std::uint8_t* sem,
                             std::size_t elems,
                             float alpha);

// =============================================================================
//! Bake a depth heat map into the frame.
// -----------------------------------------------------------------------------
//! `depth` is a device array of `elems` metres at frame resolution. Same recipe
//! as the host `colorizeDepth()`: invalid (<= 0) samples are black, the rest are
//! turned into disparity 1/d, normalised against [vmin, vmax], truncated to a
//! 0..255 index, pushed through OpenCV's COLORMAP_JET and blended 0.6 / 0.4.
//! The two percentiles that produce [vmin, vmax] stay on the host -- they are a
//! reduction, not a paint.
// =============================================================================
cudaError_t composeDepths(cudaStream_t stream,
                          std::uint8_t* bgr,
                          int width,
                          int height,
                          std::size_t stride,
                          const float* depth,
                          std::size_t elems,
                          float vmin,
                          float vmax);

}  // namespace trt_alpha::kernels::ops
