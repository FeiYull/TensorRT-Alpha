// =============================================================================
//  trt_alpha :: kernels :: preprocess
// -----------------------------------------------------------------------------
//  Preprocessing operators (CUDA). Every pointer argument refers to
//  [Device memory].
//
//  AffineMat is defined here: it is produced by the letterbox geometry
//  computation (a preprocessing-stage artifact) and consumed by
//  resizeLetterbox; postprocess only reads it to invert the transform.
// =============================================================================
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_alpha::kernels {

//! BGR HWC float (Device) -> RGB HWC float (Device), in-place.
//! No normalization, no layout change. Used by EfficientDet (NHWC input).
void bgrToRgbHwc(cudaStream_t stream, int batch,
                 float* data, int width, int height);

//! Copy float BGR HWC (Device) as-is into dst at (padTop, padLeft),
//! filling the remaining area with padValue. No scaling.
//! src size = srcW x srcH, dst size = dstW x dstH.
//! Used by YOLO-NAS: "first resize to 636x636, then pad to 640x640".
void copyWithPadding(cudaStream_t stream, int batch,
                     const float* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, int padTop, int padLeft);

//! 2x3 affine matrix (network input coords -> source image coords).
//! letterbox performs uniform scaling + translation only, so v1 == v3 == 0.
struct AffineMat
{
    float v0, v1, v2;
    float v3, v4, v5;
};

//! Fused kernel: float BGR HWC (Device) -> normalize + (optional BGR->RGB) + NCHW (Device)
//! Numeric semantics: out = (src / scale - mean) / std
//! swapRB = true performs BGR->RGB (used by YOLOv5/v6/v7/v8);
//! swapRB = false keeps the channel order (used by YOLOX).
void bgrToNchwNormalized(cudaStream_t stream, int batch,
                         const float* src, float* dst,
                         int width, int height,
                         float scale, const float mean[3], const float std_[3],
                         bool swapRB = true);

//! HWC float (Device) -> CHW float (Device). No normalization, no channel
//! swap, no color conversion.
//! Used by YuNet (after H2D conversion to float HWC, it must be reordered to NCHW).
void hwcToChw(cudaStream_t stream, int batch,
              const float* src, float* dst,
              int width, int height);

//! Divide each image by its RGB maximum value (u2net-specific).
//! maxVals: [batch] (Device; computed on Host with thrust and uploaded)
void divByMax(cudaStream_t stream, int batch,
              float* data, int width, int height, int channels,
              const float* maxVals);

//! letterbox bilinear resize: uint8 BGR HWC (Device) -> float BGR HWC (Device)
//! padValue is typically 114 (matching ultralytics).
//! dst2src: affine matrix mapping network input coords -> source image coords.
void resizeLetterbox(cudaStream_t stream, int batch,
                     const std::uint8_t* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, AffineMat dst2src);

//! float-input variant of the letterbox bilinear resize:
//! float BGR HWC (Device) -> float BGR HWC (Device).
//! Used by EfficientDet (already converted to float during H2D).
void resizeLetterbox(cudaStream_t stream, int batch,
                     const float* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, AffineMat dst2src);

//! Non-uniform scaling (no padding): src -> dst, independent x/y scaling.
//! dst2src: affine matrix mapping dst coords -> src coords.
//! mode: RGB (3 channels) / GRAY (1 channel).
//! Used by u2net (non-uniform resize, aspect ratio not preserved).
void resizeNoPadding(cudaStream_t stream, int batch,
                     const float* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     bool isGray, AffineMat dst2src);

}  // namespace trt_alpha::kernels
