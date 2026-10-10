// =============================================================================
//  trt_alpha :: kernels :: ops :: resize
// -----------------------------------------------------------------------------
//  CUDA-C reproduction of OpenCV 4.8.1's cv::resize(INTER_LINEAR).
//
//  Pure CUDA interface, no OpenCV dependency (so it can be linked straight into
//  trt_alpha-like projects).
//
//  Memory layout (same as the other trt_alpha preprocessing kernels):
//      single image  [H][W][cn], HWC interleaved
//      batch         images are packed back to back:
//                    start of image n = base + n * H * W * cn
//      src/dst are device pointers
//
//  Supported type combinations (template parameters TIn / TOut):
//      <unsigned char, unsigned char>   8U  -> 8U    fixed-point chain,
//                                                   bit-exact with cv::resize
//      <float,         float>           32F -> 32F   two float passes, numerically
//                                                   identical to cv::resize
//      <unsigned char, float>           8U  -> 32F   uchar is read and promoted to
//                                                   float, then the 32F path runs
//                                                   (reference = convertTo(32F)
//                                                   followed by resize)
//
//  cn supports only 1 / 3 (grayscale / BGR-RGB). cn=2 and cn=4 are out of scope:
//      * cn==2: OpenCV's area-fast vector path explicitly excludes 2 channels
//               (resize.cpp:2874); on an exact 2x downscale it falls back to
//               cvRound round-half-even rounding and would need its own kernel;
//      * cn==4: it only needs one more instantiation of the CN template, but is
//               not enabled here.
//
//  One internal OpenCV optimisation (it does not affect consistency, no need to
//  work around it):
//      when the downscale factor is exactly 2x in both x and y
//      (resize.cpp:3865), OpenCV rewrites INTER_LINEAR internally to the 2x2
//      INTER_AREA fast path. This is purely a performance choice: at scale=2 the
//      bilinear weights collapse to exactly 0.5/0.5, so
//      fx = (dx+0.5)*2 - 0.5 = 2dx + 0.5 -> the fractional part is always 0.5,
//      and the fixed-point chain reduces to exactly (sum + 2) >> 2 -- bit-exact
//      with area-fast's v_rshr_pack<2> (except cn==2, already excluded).
//      3x / 4x / 5x and other integer factors do not trigger the rewrite.
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace trt_alpha::kernels::ops {

//! Interpolation mode. Only Linear is implemented in this round.
enum class Interp
{
    Nearest = 0,   //!< nearest neighbour: not implemented (reserved)
    Linear  = 1,   //!< bilinear: matches cv::resize(INTER_LINEAR)
    Cubic   = 2    //!< bicubic: not implemented (reserved)
};

namespace detail
{
//! Coefficient type: the 8U path uses fixed point (short); the 32F path uses
//! float -- exactly as OpenCV does.
template <typename TOut> struct CoefType;
template <> struct CoefType<unsigned char> { using type = short; };
template <> struct CoefType<float>         { using type = float; };
}  // namespace detail

// =============================================================================
//  CvResizePlan -- precomputation separated from the hot path
// -----------------------------------------------------------------------------
//  Why this class exists (instead of a stateless function):
//      the interpolation coefficients (xofs/xalpha/yofs/ybeta) depend only on
//      (srcW, srcH, dstW, dstH) -- not on the channel count cn, not on batch. If
//      they were recomputed and re-uploaded on every call, each frame of the hot
//      loop would waste 4 cudaMalloc + 4 H2D (tens of microseconds of idle time).
//
//      This class moves "compute coefficients + allocate device memory + upload
//      once" into the constructor; launch() then issues a single kernel launch --
//      no allocation, no copy, no sync. Geometry is fixed for a video stream, so
//      the whole lifecycle constructs it once.
//
//  Usage (construct outside the loop, launch inside):
//      CvResizePlan<unsigned char, unsigned char> plan(srcW, srcH, dstW, dstH);
//      for (...) plan.launch(stream, batch, d_src, d_dst, cn);
//
//  Note: geometry (srcW/srcH/dstW/dstH) must be rebuilt when it changes; cn and
//        batch never enter the plan.
// =============================================================================
template <typename TIn, typename TOut>
class CvResizePlan
{
public:
    using CoefT = typename detail::CoefType<TOut>::type;

    //! Constructor: compute coefficients (host) + cudaMalloc + one H2D upload.
    //! Failure can be inspected via ok() / lastError().
    CvResizePlan(int srcW, int srcH, int dstW, int dstH);
    ~CvResizePlan();

    CvResizePlan(const CvResizePlan&)            = delete;
    CvResizePlan& operator=(const CvResizePlan&) = delete;
    CvResizePlan(CvResizePlan&& other) noexcept;

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

    // -------------------------------------------------------------------------
    //! Fixed-point coefficient tables (device pointers), so that fused operators
    //! can reuse the very same coefficients.
    //!
    //! A fused kernel (e.g. preprocess_fused) must be numerically bit-exact with
    //! cv::resize, and consistency depends entirely on how these four tables are
    //! built. Copying buildCoeffs elsewhere would drift sooner or later, so
    //! instead of duplicating it we expose the already-verified result here.
    //!
    //! Indexing is identical to the kernels in this file:
    //!     sx0 = xofs[dx];  a0 = xalpha[dx*2+0];  a1 = xalpha[dx*2+1];   // dx in [0, dstW)
    //!     sy0 = yofs[dy];  b0 = ybeta [dy*2+0];  b1 = ybeta [dy*2+1];   // dy in [0, dstH)
    //! The pointers stay valid for the lifetime of this object; they may be
    //! nullptr when the plan failed to construct.
    // -------------------------------------------------------------------------
    struct CoefTables
    {
        const int*   xofs;
        const CoefT* xalpha;   //!< [2*dstW]
        const int*   yofs;
        const CoefT* ybeta;    //!< [2*dstH]
    };
    CoefTables coefTables() const noexcept
    {
        return CoefTables{ d_xofs_, d_xalpha_, d_yofs_, d_ybeta_ };
    }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel.
    //!   stream : for ordering against upstream/downstream; pass 0 if unused.
    //!   batch  : number of images; src/dst must hold batch images each.
    //!   cn     : channel count, only 1 / 3.
    //! No allocation, no copy, no sync; records errors only (retrieve via lastError()).
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const TIn* src, TOut* dst, int cn,
                Interp interp = Interp::Linear) const;

private:
    void release() noexcept;

    int    srcW_ = 0, srcH_ = 0, dstW_ = 0, dstH_ = 0;
    int*   d_xofs_   = nullptr;   //!< [dstW]
    CoefT* d_xalpha_ = nullptr;   //!< [2*dstW]
    int*   d_yofs_   = nullptr;   //!< [dstH]
    CoefT* d_ybeta_  = nullptr;   //!< [2*dstH]
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  cvResize -- one-shot convenience version
// -----------------------------------------------------------------------------
//  Creates and destroys internally (so it recomputes / re-uploads coefficients
//  every call). Use it for a single call or in tests; the hot loop should use
//  CvResizePlan.
// =============================================================================
template <typename TIn, typename TOut>
void cvResize(cudaStream_t stream, int batch,
              const TIn* src, int srcW, int srcH,
              TOut* dst, int dstW, int dstH,
              int cn, Interp interp = Interp::Linear);

}  // namespace trt_alpha::kernels::ops
