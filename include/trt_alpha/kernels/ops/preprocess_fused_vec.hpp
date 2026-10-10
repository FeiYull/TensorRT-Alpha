// =============================================================================
//  trt_alpha :: kernels :: ops :: preprocess_fused_vec
// -----------------------------------------------------------------------------
//  The VECTORIZED variant of preprocessFusedDevice.
//
//  The math is unchanged -- same fixed-point chain, same coefficient tables, same
//  /255 -- and the output is bit-identical. Only the memory access pattern and the
//  amount of work per thread change:
//
//      scalar (preprocess_fused)             this file (vec)
//      ----------------------------          ----------------------------
//      1 thread = 1 output pixel             1 thread = 4 consecutive output
//                                            pixels (one quad)
//      3 scalar 4B writes per pixel          one 16B vector write per plane
//                                            (float4)
//      one integer division p/outW per pixel one per quad (4x fewer)
//      yofs/ybeta read once per pixel        reused within a quad when the row
//                                            index is the same (see below)
//      normal stores (occupy L2)             __stcs streaming stores (written
//                                            then never read; they would
//                                            otherwise evict the source image)
//
//  ----------------------------------------------------------- why it is safe
//  1) If the output plane plane = outW*outH is divisible by 4, then p0 = 4*q is
//     guaranteed 16B aligned and the float4 store is valid; otherwise
//     (plane % 4 != 0) it falls back to scalar stores automatically -- no
//     out-of-bounds access, no misalignment.
//  2) The 4 pixels of a quad are consecutive within the plane, so even across
//     rows it is still one contiguous store. The row index advances with
//     "1 division + 3 increments" instead of 4 divisions.
//  3) Each pixel still computes its own ix/iy/coefficients, exactly like the
//     scalar version -- no "shared approximation" was introduced.
//  4) The rounding, clamping, (>>4), (>>16), (+2)>>2 are all copied verbatim, and
//     BGR->RGB still works by writing position.
//
//  WARNING: the output size is fixed (static ONNX); the geometry and the
//  definition of LetterBoxInfo live in preprocess_fused.hpp.
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/preprocess_fused.hpp"   // reuse LetterBoxInfo / NormMode

#include <memory>

namespace trt_alpha::kernels::ops {

// =============================================================================
//  CvPreprocessVecPlan -- interface identical to CvPreprocessPlan, faster kernel
// -----------------------------------------------------------------------------
//  Usage:
//      auto plan = CvPreprocessVecPlan::byDivisor(500, 375, 640, 640);
//      for (...) plan.launch(stream, batch, d_src, d_dst);
// =============================================================================
class CvPreprocessVecPlan
{
public:
    CvPreprocessVecPlan(int srcW, int srcH, int outW, int outH,
                        double padValue = 114.0,
                        NormMode mode = NormMode::DivConst,
                        float a = 255.0f, float b = 0.0f);

    //! dst = src / divisor  -- bit-exact with numpy's x.astype(float32)/255.0
    static CvPreprocessVecPlan byDivisor(int srcW, int srcH, int outW, int outH,
                                         double padValue = 114.0, float divisor = 255.0f);

    //! dst = src * alpha + beta -- matches cv::Mat::convertTo's semantics
    //! (note the 1 ULP difference)
    static CvPreprocessVecPlan byAffine(int srcW, int srcH, int outW, int outH,
                                        double padValue = 114.0,
                                        float alpha = 1.0f / 255.0f, float beta = 0.0f);

    ~CvPreprocessVecPlan();

    CvPreprocessVecPlan(const CvPreprocessVecPlan&)            = delete;
    CvPreprocessVecPlan& operator=(const CvPreprocessVecPlan&) = delete;
    CvPreprocessVecPlan(CvPreprocessVecPlan&& other) noexcept;

    bool matches(int srcW, int srcH, int outW, int outH) const noexcept
    {
        return info_.srcW == srcW && info_.srcH == srcH
            && info_.outW == outW && info_.outH == outH;
    }

    const LetterBoxInfo& info() const noexcept { return info_; }

    int    outW() const noexcept { return info_.outW; }
    int    outH() const noexcept { return info_.outH; }
    size_t outElems() const noexcept { return (size_t)info_.outW * info_.outH * 3; }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    //! Hot path: issues exactly one kernel; no allocation, no copy, no sync.
    void launch(cudaStream_t stream, int batch,
                const unsigned char* src, float* dst) const;

private:
    LetterBoxInfo info_;
    std::unique_ptr<CvResizePlan<unsigned char, unsigned char>> resize_;   //!< tables only
    NormMode mode_    = NormMode::DivConst;
    float    a_       = 255.0f;
    float    b_       = 0.0f;
    float    padNorm_ = 0.0f;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  preprocessFusedVecDevice -- one-shot convenience version
// =============================================================================
void preprocessFusedVecDevice(cudaStream_t stream, int batch,
                              const unsigned char* src, float* dst,
                              int srcW, int srcH, int outW, int outH,
                              double padValue = 114.0,
                              NormMode mode = NormMode::DivConst,
                              float a = 255.0f, float b = 0.0f);

}  // namespace trt_alpha::kernels::ops
