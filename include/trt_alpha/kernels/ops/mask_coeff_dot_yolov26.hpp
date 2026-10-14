// =============================================================================
//  trt_alpha :: kernels :: ops :: mask_coeff_dot_yolov26
// -----------------------------------------------------------------------------
//  Stage 1 of the YOLOV26 instance-segmentation mask head:
//
//      logits[Y][X] = sum_k coeffs[k] * protos[k][Y][X]      (k = 0..31)
//
//  i.e. the "masks_in @ protos.view(c, -1)" line of ultralytics ops.process_mask
//  (ultralytics/utils/ops.py:544), evaluated for ONE detection at a time.
//
//  Why this operator exists at all:
//      A prototype mask is a linear model -- the logit at every prototype pixel
//      is a dot product of the 32 coefficients with the 32 protos at that pixel.
//      Doing it here (device, one pass) instead of with OpenCV on the host keeps
//      the repo's rule that a task's numerics live in kernels/ops, and it lets
//      the very same bytes flow straight into scale_mask_yolov26 (stage 2),
//      which is the already-verified ATen bilinear port.
//
//  Why per-detection and not the whole batch at once:
//      The official order is "upsample the logits, THEN binarize, THEN crop", so
//      the logits have to exist before any thresholding happens. Upsampling all
//      of a 300-detection budget at once would need ~500 MB of scratch, while a
//      real image keeps a handful of detections. One detection per launch keeps
//      the scratch at 160*160 + 640*640 floats, and the call count is bounded by
//      the number of survivors, not by top_k.
//
//  Accumulation order is ascending k, in float32. Torch's CPU GEMM blocks its
//  own accumulation differently, so the two are not bit-identical -- the same
//  caveat that applies to the semantic head. Because the value is only ever
//  compared against 0.0 afterwards, that shows up as a handful of pixels right
//  on the decision boundary (measured: <= 1e-4 % of the mask).
//
//  Conventions (same as every other operator in this directory):
//      * Plan = "an action", built once from immutable geometry.
//      * launch() issues exactly one kernel: no allocation, no copy, no sync.
//      * Nothing throws: a failure shows up as ok() == false / lastError().
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {

// =============================================================================
//! "coefficients x prototypes -> mask logits" for a single detection.
// =============================================================================
class CvMaskCoeffDotPlan
{
public:
    //! numCoeffs is the prototype count (32 for yolo26-seg), protoH/protoW the
    //! prototype resolution (160x160, i.e. 1/4 of the 640x640 network input).
    CvMaskCoeffDotPlan(int numCoeffs, int protoH, int protoW);

    [[nodiscard]] int numCoeffs() const noexcept { return numCoeffs_; }
    [[nodiscard]] int protoH()    const noexcept { return protoH_; }
    [[nodiscard]] int protoW()    const noexcept { return protoW_; }

    //! Elements of one mask.
    [[nodiscard]] std::size_t outElems() const noexcept
    {
        return static_cast<std::size_t>(protoH_) * static_cast<std::size_t>(protoW_);
    }

    [[nodiscard]] bool        ok()        const noexcept { return err_ == cudaSuccess; }
    [[nodiscard]] cudaError_t lastError() const noexcept { return err_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel. No allocation, no copy, no sync.
    //!   coeffs must hold numCoeffs floats
    //!   protos must hold numCoeffs * protoH * protoW floats, channel-major
    //!   out    must hold protoH * protoW floats and is fully overwritten
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream,
                const float* coeffs, const float* protos, float* out) const;

private:
    int numCoeffs_ = 0;
    int protoH_    = 0;
    int protoW_    = 0;

    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//! maskCoeffDotDevice -- one-shot convenience version (no plan reuse).
// =============================================================================
void maskCoeffDotDevice(cudaStream_t stream,
                        const float* coeffs, const float* protos, float* out,
                        int numCoeffs, int protoH, int protoW);

}  // namespace trt_alpha::kernels::ops
