// =============================================================================
//  bgr_to_rgba -- copy a BGR8 device image into a device RGBA8 block (GL-only)
// -----------------------------------------------------------------------------
//  The GL display backend uploads device frames through CUDA-GL interop into a
//  PIXEL_UNPACK buffer (PBO), then glTexSubImage2D() copies the PBO into the
//  texture -- both stay on the GPU, no D2H. This header exposes only the one
//  kernel that expands tightly-packed BGR8 into RGBA8; everything GL / interop
//  lives in gl_context.cpp.
//
//  The output texel is (B, G, R, 255): the BGR bytes go into the R/G/B channels
//  unchanged and alpha is set. That is exactly what the host upload path produces
//  (BGR data uploaded as GL_RGB, GL expands alpha to 1.0), so both paths feed the
//  same `.bgr` fragment-shader swizzle with no per-path branch.
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace trt_alpha::renderer {

//! Expand a tightly-packed BGR8 device image into `dst` as (B, G, R, 255) texels.
//!
//!   dst        device pointer to a w * h RGBA8 block (a mapped PBO)
//!   src        device pointer to the BGR8 frame (3 bytes per pixel)
//!   srcStride  bytes per row of `src` (== width * 3 for the decoder's tight layout)
//!   width / height  frame dimensions in pixels
//!
//! Launches on `stream` and returns cudaGetLastError() -- no synchronisation.
[[nodiscard]] cudaError_t bgr8ToRgba(cudaStream_t stream,
                                     const std::uint8_t* src,
                                     int srcStride,
                                     std::uint8_t* dst,
                                     int width,
                                     int height);

}  // namespace trt_alpha::renderer
