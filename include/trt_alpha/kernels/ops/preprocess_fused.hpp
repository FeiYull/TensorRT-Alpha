// =============================================================================
//  trt_alpha :: kernels :: ops :: preprocess_fused
// -----------------------------------------------------------------------------
//  Five-in-one: fuses the 5 YOLO preprocessing operators into a SINGLE kernel.
//
//      resize (letterbox, INTER_LINEAR)   cv::resize
//      copyMakeBorder (BORDER_CONSTANT)   cv::copyMakeBorder
//      BGR -> RGB                         cv::cvtColor(COLOR_BGR2RGB)
//      HWC -> CHW                         np.ascontiguousarray(t.transpose(2,0,1))
//      /255 (float)                       torch .float().div_(255.0)
//
//  Input : uchar*  BGR HWC   single image [H][W][3]       batch back to back
//  Output: float*  RGB CHW   single image [3][outH][outW] batch back to back
//                            (fed straight to the network)
//
//  ------------------------------------------------------------- memory access
//  Thread mapping: one thread = one output pixel (all 3 channels together)
//    the x dimension flattens the outW*outH output pixels of one image,
//    the y dimension is the batch index.
//  Writes: the ox of 32 threads within a warp are consecutive -> one 128B
//    coalesced write per plane, 3 in total.
//  Reads: each thread reads 2x2 source pixels x 3 channels = 12 bytes; when
//    upscaling, neighbouring threads often land on the same source pixel, so L1
//    hit rate is high.
//  Padding pixels (about 25% of the output area on VOC) write a constant
//    directly, without touching the source or interpolating.
//
//  Compared with "5 separate operators": 4 fewer u8 intermediate results are
//  written to memory, memory traffic drops by roughly 50-60%, and the number of
//  kernel launches goes from 5 to 1.
//
//  ----------------------------------------------- why it can still be bit-exact
//  Every output pixel needs only 2x2 source pixels, so cv::resize's fixed-point
//  chain can be reproduced in place, with no intermediate buffer:
//      horizontal: rowA = s00*a0 + s01*a1                            (int)
//      vertical:   val  = ((b0*(rowA>>4))>>16 + ((b1*(rowB>>4))>>16) + 2) >> 2
//  a0/a1/b0/b1 are taken directly from the coefficient tables inside
//  CvResizePlan that are already verified to give max=0
//  (CvResizePlan::coefTables()), so the two paths cannot drift.
//
//  ------------------------------------------------- fixed output size (letterbox)
//  The network is a static ONNX, so its input size is fixed; hence a fixed-size
//  letterbox here (equivalent to ultralytics LetterBox((outW,outH), auto=False,
//  center=True)):
//      r    = min(outW/srcW, outH/srcH)                (no clamp, small images
//                                                       are upscaled too)
//      newW = round(srcW*r)      newH = round(srcH*r)      <- round = half-to-even
//      dw   = outW - newW        dh   = outH - newH
//      padL = round(dw/2 - 0.1)  padR = round(dw/2 + 0.1)
//      padT = round(dh/2 - 0.1)  padB = round(dh/2 + 0.1)
//  so padL+padR = dw, padT+padB = dh and the output is always exactly outW x outH.
//
//  WARNING: this differs from the auto=True/rect letterbox used in some Python
//  side pipelines: that one produces an output that is a multiple of 32 (on VOC
//  almost always 640x480 / 640x448) with a non-fixed size, which cannot feed a
//  static engine.
//
//  ------------------------------------------------ reverse mapping (for bboxes)
//  Map a bbox back to the original image after inference:
//      x_orig = (x_net - padLeft) / ratio
//      y_orig = (y_net - padTop ) / ratio
//  These values live in Plan::info() (a POD struct, since they are still needed
//  after inference).
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/resize.hpp"     // reuse CvResizePlan's fixed-point tables
#include "trt_alpha/kernels/ops/normalize.hpp"  // reuse NormMode (DivConst / Affine)
#include "trt_alpha/kernels/ops/numeric.hpp"

#include <memory>

namespace trt_alpha::kernels::ops {

// =============================================================================
//  LetterBoxInfo -- geometry + reverse mapping parameters (POD, still needed on
//  the inference side)
// =============================================================================
struct LetterBoxInfo
{
    int    srcW    = 0, srcH    = 0;   //!< original image size
    int    newW    = 0, newH    = 0;   //!< after uniform scaling (before padding)
    int    outW    = 0, outH    = 0;   //!< network input = this operator's output
                                       //!< (fixed size)
    int    padLeft = 0, padTop  = 0;
    int    padRight = 0, padBottom = 0;
    double ratio   = 1.0;              //!< scale factor r = min(outW/srcW, outH/srcH)

    //! network coords -> original image coords (bbox reverse mapping)
    double toOrigX(double xNet) const { return (xNet - padLeft) / ratio; }
    double toOrigY(double yNet) const { return (yNet - padTop) / ratio; }
};

// =============================================================================
//! Fixed-size letterbox geometry -- the SINGLE source of truth.
//!
//! Both CvPreprocessPlan and CvPreprocessVecPlan use this function. The
//! standalone project used to keep a second, copy-pasted copy inside the vector
//! implementation that had to be kept textually identical; that duplicate is gone,
//! so the two can no longer drift.
//!
//! Implemented in preprocess_fused.cu.
// =============================================================================
LetterBoxInfo computeLetterBox(int srcW, int srcH, int outW, int outH);

// =============================================================================
//  CvPreprocessPlan -- geometry fixed in the constructor, only launch in the loop
// -----------------------------------------------------------------------------
//  The constructor does three things: compute the letterbox geometry, build the
//  internal CvResizePlan (computes the fixed-point coefficients and uploads them)
//  and convert the fill value into a normalized constant. Afterwards launch()
//  neither allocates, nor copies, nor syncs.
//
//  Images in one batch must share geometry (same srcW/srcH -> same outW/outH);
//  rebuild the plan when the geometry changes.
//  Usage:
//      auto plan = CvPreprocessPlan::byDivisor(500, 375, 640, 640);   // fixed 640x640
//      for (...) plan.launch(stream, batch, d_src, d_dst);            // d_dst is float*
// =============================================================================
class CvPreprocessPlan
{
public:
    //! Explicit constructor. padValue is the fill value in the u8 domain
    //! (default 114, as YOLO uses): the reference chain "fills u8 first, then
    //! divides by 255", so this does saturate_cast<uchar> first and normalizes
    //! afterwards.
    CvPreprocessPlan(int srcW, int srcH, int outW, int outH,
                     double padValue = 114.0,
                     NormMode mode = NormMode::DivConst,
                     float a = 255.0f, float b = 0.0f);

    //! dst = src / divisor     -- default 255, bit-exact with numpy's
    //! x.astype(float32)/255.0
    static CvPreprocessPlan byDivisor(int srcW, int srcH, int outW, int outH,
                                      double padValue = 114.0, float divisor = 255.0f);

    //! dst = src * alpha + beta -- matches cv::Mat::convertTo's semantics
    //! (note the 1 ULP difference)
    static CvPreprocessPlan byAffine(int srcW, int srcH, int outW, int outH,
                                     double padValue = 114.0,
                                     float alpha = 1.0f / 255.0f, float beta = 0.0f);

    ~CvPreprocessPlan();

    CvPreprocessPlan(const CvPreprocessPlan&)            = delete;
    CvPreprocessPlan& operator=(const CvPreprocessPlan&) = delete;
    CvPreprocessPlan(CvPreprocessPlan&& other) noexcept;

    //! Does the geometry match this plan? (A mismatch must be rebuilt.)
    bool matches(int srcW, int srcH, int outW, int outH) const noexcept
    {
        return info_.srcW == srcW && info_.srcH == srcH
            && info_.outW == outW && info_.outH == outH;
    }

    //! Geometry + reverse mapping parameters (for mapping bboxes back).
    const LetterBoxInfo& info() const noexcept { return info_; }

    int    outW() const noexcept { return info_.outW; }
    int    outH() const noexcept { return info_.outH; }
    size_t outElems() const noexcept        //!< output elements per image = 3*outW*outH
    {
        return (size_t)info_.outW * info_.outH * 3;
    }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel.
    //!   batch : number of images; src must hold batch images of srcW*srcH*3
    //!           uchar, dst must hold batch * 3*outW*outH floats.
    //! No allocation, no copy, no sync.
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const unsigned char* src, float* dst) const;

private:
    LetterBoxInfo info_;
    //! Only its fixed-point coefficient tables are borrowed (CvResizePlan is not
    //! assignable, hence the unique_ptr).
    std::unique_ptr<CvResizePlan<unsigned char, unsigned char>> resize_;
    NormMode mode_    = NormMode::DivConst;
    float    a_       = 255.0f;   //!< DivConst: divisor; Affine: alpha
    float    b_       = 0.0f;     //!< Affine only
    float    padNorm_ = 0.0f;     //!< the padding pixel value after normalization
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  preprocessFusedDevice -- one-shot convenience version
// -----------------------------------------------------------------------------
//  Creates and destroys internally (recomputes geometry and re-uploads
//  coefficients every call). Use it for a single call or in tests; the hot loop
//  should use CvPreprocessPlan.
//
//  The default is equivalent to dst = src / 255, bit-exact with numpy's
//  x.astype(float32) / 255.0.
//  It does NOT match cv::Mat::convertTo(CV_32F, 1.0/255.0) (that one uses a float
//  multiply and is 1 ULP away).
// =============================================================================
void preprocessFusedDevice(cudaStream_t stream, int batch,
                           const unsigned char* src, float* dst,
                           int srcW, int srcH, int outW, int outH,
                           double padValue = 114.0,
                           NormMode mode = NormMode::DivConst,
                           float a = 255.0f, float b = 0.0f);

}  // namespace trt_alpha::kernels::ops
