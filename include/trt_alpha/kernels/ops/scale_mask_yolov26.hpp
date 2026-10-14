// =============================================================================
//  trt_alpha :: kernels :: ops :: scale_mask_yolov26
// -----------------------------------------------------------------------------
//  Inverse letterbox for a DENSE output (depth map / semantic class map):
//  crop the letterbox padding away and resample onto the source image.
//
//  This is a line-by-line port of ultralytics `ops.scale_masks`
//  (ultralytics/utils/ops.py) with mode="bilinear" (depth) or "nearest" (sem).
//  The reference (padding=True, ratio_pad=None) is:
//
//      gain_h = gain_w = min(maskH/srcH, maskW/srcW)        # h FIRST
//      pad_w = (maskW - round(srcW*gain_w)) / 2
//      pad_h = (maskH - round(srcH*gain_h)) / 2
//      top, left     = round(pad_h - 0.1), round(pad_w - 0.1)
//      bottom, right = top + round(srcH*gain_h), left + round(srcW*gain_w)
//      out = F.interpolate(mask[..., top:bottom, left:right], (srcH, srcW), mode=...)
//
//  (top, left, round(srcW*gain), round(srcH*gain)) is EXACTLY the letterbox
//  geometry of computeLetterBox() (preprocess_fused_yolov26_det), and this Plan
//  calls that very function, so the forward and the inverse transform cannot
//  drift apart.
//
//  ------------------------------------------------------- why this exact form
//  The bilinear branch MUST be written as
//      h0l * (w0l*v00 + w1l*v01) + h1l * (w0l*v10 + w1l*v11)
//  and NOT as the algebraically equivalent
//      top = v00 + fx*(v01-v00); bot = v10 + fx*(v11-v10); out = top + fy*(bot-top)
//  Torch's CUDA kernel uses the first form; the second rounds differently in
//  float32 and was measured to differ by 1.4e-6 (see sry_probe_scale_mask.py).
//
//  Interpolation is align_corners=False (torch's default):
//      rh   = (float)inH / (float)outH                    <- float division
//      src  = rh * (dst + 0.5f) - 0.5f, clamped to >= 0
//      i0   = (int)src ; i1 = min(i0+1, inH-1)
//      w1   = src - i0 ; w0 = 1 - w1
//  nearest uses torch's non-exact kernel:
//      src = min((int)floorf((float)dst * rh), inH - 1)
//
//  Conventions (same as the rest of kernels/ops):
//      * never throws, never syncs -- errors land in err_ (query ok()/lastError())
//      * layout [batch][channels][H][W]
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/preprocess_fused_yolov26_det.hpp"   // computeLetterBox

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {

//! How the cropped mask is resampled onto the source image.
//!  * Bilinear -- depth regression (float only)
//!  * Nearest  -- class ids (uint8); bilinear would invent classes
enum class MaskInterp
{
    Bilinear,
    Nearest,
};

// =============================================================================
//  ScaleMaskInfo -- the crop window + output size (POD, still useful after the
//  call for debugging)
// =============================================================================
struct ScaleMaskInfo
{
    int srcW = 0, srcH = 0;      //!< output = the original image size
    int netW = 0, netH = 0;      //!< input = the engine's dense-output size
    int newW = 0, newH = 0;      //!< letterbox content size inside the mask
    int cropLeft = 0, cropTop = 0;
};

// =============================================================================
//  CvScaleMaskPlan -- geometry fixed in the constructor, only launch in the loop
// -----------------------------------------------------------------------------
//  Usage:
//      CvScaleMaskPlan<float> plan(500, 375, 640, 640, MaskInterp::Bilinear);
//      for (...) plan.launch(stream, batch, d_mask, d_depth, 1);
// =============================================================================
template <typename T>
class CvScaleMaskPlan
{
public:
    //! padding == true mirrors scale_masks(padding=True), the only mode used by
    //! ultralytics' depth / semantic postprocessing.
    CvScaleMaskPlan(int srcW, int srcH, int netW, int netH,
                    MaskInterp mode = MaskInterp::Bilinear,
                    bool padding = true);

    //! Does the geometry match this plan? (A mismatch must be rebuilt.)
    bool matches(int srcW, int srcH, int netW, int netH) const noexcept
    {
        return info_.srcW == srcW && info_.srcH == srcH
            && info_.netW == netW && info_.netH == netH;
    }

    [[nodiscard]] const ScaleMaskInfo& info() const noexcept { return info_; }
    [[nodiscard]] MaskInterp mode() const noexcept { return mode_; }

    [[nodiscard]] int outW() const noexcept { return info_.srcW; }
    [[nodiscard]] int outH() const noexcept { return info_.srcH; }

    //! Output elements per channel.
    [[nodiscard]] std::size_t outElems() const noexcept
    {
        return static_cast<std::size_t>(info_.srcW) * info_.srcH;
    }

    [[nodiscard]] bool        ok()        const noexcept { return err_ == cudaSuccess; }
    [[nodiscard]] cudaError_t lastError() const noexcept { return err_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel. No allocation, no copy, no sync.
    //!   src must hold batch*channels*netW*netH elements
    //!   dst must hold batch*channels*srcW*srcH elements
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const T* src, T* dst, int channels = 1) const;

private:
    ScaleMaskInfo info_{};
    MaskInterp    mode_ = MaskInterp::Bilinear;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  scaleMaskDevice -- one-shot convenience version (rebuilds geometry per call)
// =============================================================================
template <typename T>
void scaleMaskDevice(cudaStream_t stream, int batch,
                     const T* src, T* dst, int channels,
                     int srcW, int srcH, int netW, int netH,
                     MaskInterp mode = MaskInterp::Bilinear);

extern template class CvScaleMaskPlan<float>;
extern template class CvScaleMaskPlan<unsigned char>;

}  // namespace trt_alpha::kernels::ops
