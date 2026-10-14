// =============================================================================
//  trt_alpha :: kernels :: ops :: pil_resample
// -----------------------------------------------------------------------------
//  CUDA-C reproduction of PIL / libImaging's `Image.resize(size, BILINEAR)` --
//  which is the resampler torchvision's `transforms.Resize` actually runs.
//
//  ------------------------------------------------------- why not cv::resize
//  torchvision's `Resize` receives a PIL image and hands the geometry straight
//  to Pillow (torchvision/transforms/_functional_pil.py: `img.resize(size[::-1],
//  interpolation)`), so the code that runs is
//      pillow-12.3.0/src/libImaging/Resample.c
//  not OpenCV's. The two differ by more than rounding: Pillow scales the filter
//  support when downscaling
//      filterscale = max(1, scale);  support = filter_support * filterscale
//  i.e. it ANTIALIASES, while cv::resize(INTER_LINEAR) keeps a fixed 2x2 kernel.
//  Measured on this machine 1280x1280 -> 640x640 differs by up to 34/255 on
//  30-56% of the pixels, so a cv2-based "approximation" changes the network
//  input outright. Hence this operator.
//
//  --------------------------------------------------- why the port is bit-exact
//  Every arithmetic step of Pillow's 8-bit path is integer:
//      1. double weights, normalized so that they sum to 1
//      2. quantized once to INT32 fixed point, PRECISION_BITS = 22, with a
//         sign-aware +/-0.5 offset (truncated, not "round()")
//      3. accumulate `value * coef` starting from 1 << 21, then
//         clip8(v) = clamp(v >> 22, 0, 255)      (>> being an arithmetic shift)
//  Because the accumulation is integer, the order of the taps cannot change the
//  result -- which is what makes the vectorized fused variant bit-identical by
//  construction rather than by luck. Only two things must be reproduced exactly:
//  the double->int32 quantization and clip8.
//
//  ------------------------------------------------- no intermediate is needed
//  Pillow runs two passes and clips the horizontal result to uint8 before the
//  vertical pass reads it (a lossy intermediate buffer). This operator computes
//  the same numbers in ONE kernel with no buffer: for every vertical tap row it
//  computes the horizontal value (including its clip8) and accumulates it. Same
//  integers, same clip8, therefore the same result.
//
//  A pass whose output extent equals its input extent is skipped by Pillow
//  (`needH` / `needV` below): that axis is then a plain copy with NO filtering
//  and NO clip8.
//
//  Memory layout (same as the other preprocessing operators):
//      single image  [H][W][3]  BGR or RGB -- the resampler is channel agnostic,
//                               it treats the 3 bytes as one band-triple
//      batch         images are packed back to back
//      src / dst are device pointers
//
//  Only cn == 3 is supported: the ported branch is Pillow's `bands == 3`.
//  Any other cn records cudaErrorInvalidValue (explicit failure, never a silent
//  wrong result).
//
//  Error handling follows numeric.hpp: never throws, never syncs; records err_
//  only, query it with ok() / lastError().
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace trt_alpha::kernels::ops {

// -----------------------------------------------------------------------------
//  Pillow's fixed-point precision:  PRECISION_BITS = 32 - 8 - 2  (Resample.c:93)
// -----------------------------------------------------------------------------
inline constexpr int          kPilPrecisionBits = 22;
inline constexpr std::int32_t kPilCoefScale     = 1 << kPilPrecisionBits;        // 4194304
inline constexpr std::int32_t kPilRoundHalf     = 1 << (kPilPrecisionBits - 1);  // 2097152

// -----------------------------------------------------------------------------
//  PilBound -- "first source index" + "number of taps" for one output index of
//  one axis.  Mirrors Pillow's `bounds[]` array (Resample.c:261).
// -----------------------------------------------------------------------------
struct PilBound
{
    int first = 0;
    int count = 0;
};

// -----------------------------------------------------------------------------
//  PilResampleAxis -- the fixed-point coefficient table of ONE axis.
//
//  Ported from Pillow 12.3.0 src/libImaging/Resample.c:
//      precompute_coeffs      (line 182)   double weights normalized by their sum
//      normalize_coeffs_8bpc  (line 269)   quantize to INT32 (PRECISION_BITS)
// -----------------------------------------------------------------------------
struct PilResampleAxis
{
    int ksize = 0;                          //!< max number of taps per output index
    std::vector<PilBound>     bounds;       //!< [outSize]
    std::vector<std::int32_t> coef;         //!< [outSize * ksize]

    bool empty() const noexcept { return bounds.empty(); }
};

// =============================================================================
//! torchvision `_compute_resized_output_size(..., size=[n])` -- the "shortest
//! edge" mode used by `transforms.Resize(n)` on PIL images.
//!
//!     new_short = size
//!     new_long  = int(size * long / short)        <- int() truncates
//!     new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
//!
//! Exported (rather than kept private to the fused operator) because BOTH the
//! fused preprocessing and any caller of this operator need the same geometry,
//! and a second copy would drift.
//!
//! Implemented in pil_resample.cu.
// =============================================================================
void computePilResizeShortestEdge(int srcW, int srcH, int size, int& newW, int& newH);

// =============================================================================
//  CvPilResamplePlan -- precomputation separated from the hot path
// -----------------------------------------------------------------------------
//  The coefficient tables depend only on (srcW, srcH, dstW, dstH); computing and
//  uploading them per frame would waste 4 cudaMalloc + 4 H2D each time. So the
//  constructor builds them once and launch() issues a single kernel launch -- no
//  allocation, no copy, no sync.
//
//  Usage:
//      CvPilResamplePlan plan(srcW, srcH, newW, newH);
//      for (...) plan.launch(stream, batch, d_src, d_dst);
// =============================================================================
class CvPilResamplePlan
{
public:
    //! Geometry = "resize from srcW x srcH to dstW x dstH".  Failure is reported
    //! via ok() / lastError(), not by throwing.
    CvPilResamplePlan(int srcW, int srcH, int dstW, int dstH);
    ~CvPilResamplePlan();

    CvPilResamplePlan(const CvPilResamplePlan&)            = delete;
    CvPilResamplePlan& operator=(const CvPilResamplePlan&) = delete;
    CvPilResamplePlan(CvPilResamplePlan&& other) noexcept;

    //! Does the geometry match this plan? (A mismatch must be rebuilt, otherwise
    //! the result is wrong.)
    bool matches(int srcW, int srcH, int dstW, int dstH) const noexcept
    {
        return srcW_ == srcW && srcH_ == srcH && dstW_ == dstW && dstH_ == dstH;
    }

    int srcW() const noexcept { return srcW_; }
    int srcH() const noexcept { return srcH_; }
    int dstW() const noexcept { return dstW_; }
    int dstH() const noexcept { return dstH_; }

    bool needH() const noexcept { return needH_; }   //!< dstW != srcW -> pass runs
    bool needV() const noexcept { return needV_; }   //!< dstH != srcH -> pass runs
    int  ksizeH() const noexcept { return axisH_.ksize; }
    int  ksizeV() const noexcept { return axisV_.ksize; }

    // -------------------------------------------------------------------------
    //! Fixed-point coefficient tables (device pointers), so that fused operators
    //! can reuse the very same coefficients instead of rebuilding them.
    //!
    //! Indexing is identical to the kernels in this file:
    //!     b = boundsH[dx];  k = coefH + dx * ksizeH;   // dx in [0, dstW)
    //!     b = boundsV[dy];  k = coefV + dy * ksizeV;   // dy in [0, dstH)
    //! The pointers stay valid for the lifetime of this object; they are nullptr
    //! when the plan failed to construct.
    // -------------------------------------------------------------------------
    struct CoefTables
    {
        const PilBound*     boundsH;
        const std::int32_t* coefH;
        int  ksizeH;
        const PilBound*     boundsV;
        const std::int32_t* coefV;
        int  ksizeV;
        int  needH;   //!< 0 / 1, as the fused kernels expect
        int  needV;
    };
    CoefTables coefTables() const noexcept
    {
        return CoefTables{ d_boundsH_, d_coefH_, axisH_.ksize,
                           d_boundsV_, d_coefV_, axisV_.ksize,
                           needH_ ? 1 : 0, needV_ ? 1 : 0 };
    }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel.
    //!   batch : number of images; src/dst must hold batch images each.
    //!   cn    : channel count, only 3 (see the header comment)
    //! No allocation, no copy, no sync.
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const unsigned char* src, unsigned char* dst, int cn = 3) const;

private:
    void release() noexcept;

    int  srcW_ = 0, srcH_ = 0, dstW_ = 0, dstH_ = 0;
    bool needH_ = false, needV_ = false;

    PilResampleAxis axisH_;   //!< empty when needH_ is false
    PilResampleAxis axisV_;   //!< always built (its geometry sizes the output)

    PilBound*     d_boundsH_ = nullptr;   //!< [dstW]
    std::int32_t* d_coefH_   = nullptr;   //!< [dstW * ksizeH]
    PilBound*     d_boundsV_ = nullptr;   //!< [dstH]
    std::int32_t* d_coefV_   = nullptr;   //!< [dstH * ksizeV]

    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  pilResampleDevice -- one-shot convenience version
// -----------------------------------------------------------------------------
//  Creates and destroys internally (so it rebuilds and re-uploads the tables
//  every call). Use it for a single call or in tests; the hot loop should use
//  CvPilResamplePlan.
// =============================================================================
void pilResampleDevice(cudaStream_t stream, int batch,
                       const unsigned char* src, unsigned char* dst,
                       int srcW, int srcH, int dstW, int dstH, int cn = 3);

}  // namespace trt_alpha::kernels::ops
