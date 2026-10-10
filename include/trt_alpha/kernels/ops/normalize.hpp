// =============================================================================
//  trt_alpha :: kernels :: ops :: normalize
// -----------------------------------------------------------------------------
//  Normalization / linear scaling: dst = src <op> constant. Two modes.
//
//  ------------------------------------------------------ why the default is division
//  Measured (walking all 256 uchar values, float32):
//
//      (float)x / 255.0f                    -> baseline (= numpy's x.astype(float32)/255.0)
//      (float)x * (1.0f/255.0f)             -> 126/256 values differ by 1 ULP     X
//      (double)x * (1.0/255.0) -> (float)   -> bit-identical with the baseline     O
//
//  In other words "divide by 255" and "multiply by 1/255" are equal in the reals
//  but not under IEEE754 -- the latter rounds 1/255 to float32 first and loses
//  1 ULP on the way back.
//
//  A point that is very easy to get wrong (measured by test_cv_ops, 700 configurations):
//
//      cv::Mat::convertTo(CV_32F, 1.0/255.0)  vs  division          -> 1 ULP apart
//      cv::Mat::convertTo(CV_32F, 1.0/255.0)  vs  float multiply    -> bit-identical
//
//  That is, OpenCV's convertTo (the 8U->32F SIMD path) uses a float multiply, and
//  it is simply not the same number as numpy's `x / 255.0`. So "which reference am
//  I matching?" must be answered first:
//      * reference is numpy  x.astype(float32) / 255.0       -> use DivConst
//      * reference is cv::Mat::convertTo(CV_32F, 1.0/255.0)  -> use byAffine(1.0f/255.0f, 0)
//
//  Both are provided here; the default is DivConst (numpy semantics), and the
//  kernel explicitly uses __fdiv_rn (IEEE correctly-rounded division) to guarantee
//  bit-exactness.
//
//  ------------------------------------------------------------------- two modes
//      DivConst :  dst = src / divisor         default divisor = 255; bit-exact path
//      Affine   :  dst = src * alpha + beta    general path, semantics match
//                                              cv::Mat::convertTo's alpha/beta
//                                              (alpha taken from a float literal)
//
//  Supported type combinations (TIn, TOut):
//      <unsigned char, float>   8U  -> 32F     (uchar widens to float losslessly)
//      <float,         float>   32F -> 32F
//
//  Element-wise operator, independent of geometry; launch() takes cn and hw
//  (elements per image = cn * hw).
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/numeric.hpp"

namespace trt_alpha::kernels::ops {

//! Normalization mode.
enum class NormMode
{
    DivConst = 0,   //!< dst = src / divisor         -- default, bit-exact path
    Affine   = 1    //!< dst = src * alpha + beta    -- general path
};

// =============================================================================
//  CvNormalizePlan
// -----------------------------------------------------------------------------
//  Static factories are used instead of a single-argument constructor to avoid
//  ambiguity about whether "CvNormalizePlan(1.0f/255.0f)" is a divisor or a
//  multiplier:
//
//      CvNormalizePlan<unsigned char, float> plan;                       // /255
//      CvNormalizePlan<unsigned char, float> plan = CvNormalizePlan<unsigned char, float>::byDivisor(255.0f);
//      auto plan = CvNormalizePlan<unsigned char, float>::byAffine(1.f/255.f, 0.f);
//
//      for (...) plan.launch(stream, batch, d_src, d_dst, cn, hw);
// =============================================================================
template <typename TIn, typename TOut>
class CvNormalizePlan
{
public:
    //! Default: DivConst with divisor = 255.0f (the most common /255 normalization).
    CvNormalizePlan();

    //! dst = src / divisor        -- bit-exact path (divisor must not be 0)
    static CvNormalizePlan byDivisor(float divisor);

    //! dst = src * alpha + beta   -- general path, semantics match cv::Mat::convertTo
    static CvNormalizePlan byAffine(float alpha, float beta);

    ~CvNormalizePlan();

    CvNormalizePlan(const CvNormalizePlan&)            = delete;
    CvNormalizePlan& operator=(const CvNormalizePlan&) = delete;
    CvNormalizePlan(CvNormalizePlan&& other) noexcept;

    //! Note: this compares floats with ==, only to decide whether the current plan
    //! can be reused; the coefficients passed in should be bit-identical literals
    //! to the ones used at construction.
    bool matches(NormMode mode, float a, float b) const noexcept
    {
        return mode_ == mode && a_ == a && b_ == b;
    }

    NormMode mode() const noexcept { return mode_; }
    float    a()    const noexcept { return a_; }   //!< DivConst: divisor; Affine: alpha
    float    b()    const noexcept { return b_; }   //!< used by Affine only

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel (1D grid, element-wise).
    //!   batch : number of images
    //!   cn    : channel count, only 1 / 3
    //!   hw    : pixels per image (H * W); total elements = batch * cn * hw
    //! No allocation, no copy, no sync.
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const TIn* src, TOut* dst, int cn, int hw) const;

private:
    NormMode mode_ = NormMode::DivConst;
    float    a_    = 255.0f;
    float    b_    = 0.0f;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  normalizeDevice -- one-shot convenience version
// -----------------------------------------------------------------------------
//  Default is equivalent to dst = src / 255, bit-exact with numpy's
//  x.astype(float32) / 255.0.
//
//  It does NOT match cv::Mat::convertTo(CV_32F, 1.0/255.0) (that one uses a float
//  multiply and is 1 ULP away). To be bit-exact with convertTo, use
//  byAffine(1.0f/255.0f, 0.0f). See the top of this file for details.
// =============================================================================
template <typename TIn, typename TOut>
void normalizeDevice(cudaStream_t stream, int batch,
                     const TIn* src, TOut* dst,
                     int cn, int hw,
                     NormMode mode = NormMode::DivConst,
                     float a = 255.0f, float b = 0.0f);

}  // namespace trt_alpha::kernels::ops
