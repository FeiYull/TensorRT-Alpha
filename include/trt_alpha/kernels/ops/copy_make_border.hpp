// =============================================================================
//  trt_alpha :: kernels :: ops :: copy_make_border
// -----------------------------------------------------------------------------
//  CUDA-C reproduction of OpenCV 4.8.1's
//  cv::copyMakeBorder(..., BORDER_CONSTANT, value).
//
//  OpenCV reference implementation (modules/core/src/copy.cpp :: cv::copyMakeBorder):
//
//      if( borderType == BORDER_CONSTANT )
//      {
//          dst.setTo( value );                                     // (1) fill the whole
//                                                                  //     block with value
//          src.copyTo( dst( Rect(left, top, width, height) ) );     // (2) copy src into
//                                                                  //     the centre rect
//      }
//
//  This implementation is equivalent pixel by pixel:
//      inside the centre rect (left <= x < left+srcW and top <= y < top+srcH)
//                                                        -> copy from src
//      otherwise                                         -> write value[c]
//
//  Numerical consistency: both steps are plain assignments with no arithmetic, so
//  8U and 32F must both be bit-exact (the test requires max = 0).
//
//  "value" is a double[cn] (matching OpenCV's Scalar). The host side converts it
//  to the target pixel type before launching:
//      8U  -> cvRound round-half-even + clamp to [0,255]
//             (equivalent to saturate_cast<uchar>)
//      32F -> plain (float)v
//
//  Only BORDER_CONSTANT is implemented. The other enum values are placeholders;
//  passing one returns cudaErrorNotSupported.
//
//  Memory layout (same as resize): [batch][H][W][cn], src/dst are device
//  pointers.
//  Output size: dstW = srcW + left + right, dstH = srcH + top + bottom.
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/numeric.hpp"

namespace trt_alpha::kernels::ops {

//! Border extrapolation mode. Only Constant is implemented in this round.
enum class BorderType
{
    Constant   = 0,   //!< constant fill: matches cv::BORDER_CONSTANT (implemented)
    Replicate  = 1,   //!< replicate edge: TODO (cv::BORDER_REPLICATE)
    Reflect    = 2,   //!< reflect with border: TODO (cv::BORDER_REFLECT)
    Reflect101 = 4,   //!< reflect without border: TODO (cv::BORDER_REFLECT_101)
    Wrap       = 3    //!< wrap around: TODO (cv::BORDER_WRAP)
};

// =============================================================================
//  CvCopyMakeBorderPlan
// -----------------------------------------------------------------------------
//  This operator has no coefficients to precompute and no device buffer
//  (BORDER_CONSTANT is a pure constant and needs no lookup table). The Plan is
//  kept so the usage pattern matches CvResizePlan: geometry is fixed in the
//  constructor, the loop only launches, and illegal geometry can be rejected
//  with an early return at construction time.
// =============================================================================
template <typename TIn, typename TOut>
class CvCopyMakeBorderPlan
{
public:
    CvCopyMakeBorderPlan(int srcW, int srcH, int top, int bottom, int left, int right);
    ~CvCopyMakeBorderPlan();

    CvCopyMakeBorderPlan(const CvCopyMakeBorderPlan&)            = delete;
    CvCopyMakeBorderPlan& operator=(const CvCopyMakeBorderPlan&) = delete;
    CvCopyMakeBorderPlan(CvCopyMakeBorderPlan&& other) noexcept;

    //! Does the geometry match this plan? (A mismatch must be rebuilt.)
    bool matches(int srcW, int srcH, int top, int bottom, int left, int right) const noexcept
    {
        return srcW_ == srcW && srcH_ == srcH
            && top_ == top && bottom_ == bottom && left_ == left && right_ == right;
    }

    int srcW() const noexcept { return srcW_; }
    int srcH() const noexcept { return srcH_; }
    int dstW() const noexcept { return srcW_ + left_ + right_; }
    int dstH() const noexcept { return srcH_ + top_ + bottom_; }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel. No allocation, no copy, no sync.
    //!   batch : number of images; src/dst must hold batch images each.
    //!   cn    : channel count, only 1 / 3.
    //!   value : host array of length cn, per-channel fill value; nullptr = all 0.
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const TIn* src, TOut* dst, int cn,
                BorderType borderType = BorderType::Constant,
                const double* value = nullptr) const;

private:
    int srcW_ = 0, srcH_ = 0, top_ = 0, bottom_ = 0, left_ = 0, right_ = 0;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  copyMakeBorderDevice -- one-shot convenience version
// -----------------------------------------------------------------------------
//  Creates and destroys internally. Use it for a single call or in tests; the
//  hot loop should use CvCopyMakeBorderPlan.
// =============================================================================
template <typename TIn, typename TOut>
void copyMakeBorderDevice(cudaStream_t stream, int batch,
                          const TIn* src, TOut* dst,
                          int srcW, int srcH, int cn,
                          int top, int bottom, int left, int right,
                          BorderType borderType = BorderType::Constant,
                          const double* value = nullptr);

}  // namespace trt_alpha::kernels::ops
