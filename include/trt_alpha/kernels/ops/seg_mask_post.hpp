// =============================================================================
//  trt_alpha :: kernels :: ops :: seg_mask_post
// -----------------------------------------------------------------------------
//  Stage 3 (and the small tail of stage 4) of the YOLOV26/YOLOV8 instance-seg
//  mask head -- the part that used to run as a scalar loop on the host.
//
//      up    = F.interpolate(logits, (netH, netW), mode="bilinear")   [stage 2]
//      mask  = up > 0                                                 [stage 3]
//      mask  = crop_mask(mask, boxes)      x >= x1 && x < x2 && ...
//      keep  = mask.amax() > 0
//
//  Binarize and crop are one pass here: outside the box the output is 0 anyway,
//  so "zero the plane, then compare inside the window" collapses into a single
//  store per pixel. The same pass also rewrites `up` in place as 0.0f / 1.0f --
//  that is exactly the float form of the python `bool` mask, which is what
//  ops.scale_masks() interpolates for Segmentation::maskOrig. Doing it here
//  costs no extra traffic (the value is already in a register) and saves a
//  second full-plane pass later.
//
//  Why a free function instead of the usual CvXxxPlan:
//      a Plan earns its keep when there are coefficients that depend only on
//      geometry, so they can be computed / uploaded once. Here the only fixed
//      quantity is (netW, netH); everything else is per-instance data, so there
//      is nothing to amortise and a stateless launch is the honest shape.
//
//  Conventions (same as every other operator in this directory):
//      * never throws, never syncs -- the return value is the only error channel
//      * layout: instance-major, one contiguous [netH][netW] plane per instance
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace trt_alpha::kernels::ops {

// =============================================================================
//! "upsampled logits -> binary cropped mask + foreground count", per instance.
// -----------------------------------------------------------------------------
//!  up      : [n][netH*netW] float, the stage-2 output. Overwritten IN PLACE with
//!            0.0f / 1.0f (the binarized mask in float form).
//!  upStep  : elements between two instances of `up` (normally netW*netH, but the
//!            caller stages instances inside a bigger scratch buffer).
//!  boxes   : boxes[j*boxStride + 0..3] = x1, y1, x2, y2 in NETWORK pixels, i.e.
//!            the rows the fused det postprocess already wrote.
//!  out     : [n][netH*netW] uint8, 255 where the mask is on, 0 everywhere else
//!            (including outside the box).
//!  counts  : [n] int32, foreground pixel count of instance j. The caller must
//!            zero it before the launch -- the kernel only accumulates.
//!
//!  The box window is the half-open integer range
//!      [ceil(x1), ceil(x2)) x [ceil(y1), ceil(y2))
//!  which is exactly `x >= x1 && x < x2` over integer pixel indices (right and
//!  bottom strict, as in ultralytics crop_mask).
// =============================================================================
cudaError_t binarizeCropMasks(cudaStream_t stream,
                              int n,
                              float* up,
                              std::size_t upStep,
                              const float* boxes,
                              int boxStride,
                              int netW,
                              int netH,
                              std::uint8_t* out,
                              int* counts,
                              float thr);

// =============================================================================
//! dst = (src > thr) ? 255 : 0  -- the "> 0.5" that follows scale_masks().
// -----------------------------------------------------------------------------
//! Both pointers are device pointers; `elems` is the total element count (a
//! single plane, or a whole batch of contiguous planes).
// =============================================================================
cudaError_t thresholdMaskToU8(cudaStream_t stream,
                              const float* src,
                              std::uint8_t* dst,
                              std::size_t elems,
                              float thr);

// =============================================================================
//! dst = (uint8)min(max(src, 0), 255) -- float plane to an 8-bit mask, the same
//! clamp+truncate the CPU path did per pixel. Used by the fully convolutional
//! seg heads (u2net), whose output is already a 0..255 saliency map.
// -----------------------------------------------------------------------------
//! Both pointers are device pointers; `elems` is the total element count.
// =============================================================================
cudaError_t clampToU8(cudaStream_t stream,
                      const float* src,
                      std::uint8_t* dst,
                      std::size_t elems);

}  // namespace trt_alpha::kernels::ops
