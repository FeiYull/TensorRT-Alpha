// =============================================================================
//  trt_alpha :: kernels :: ops :: preprocess_fused_vec_yolov26_cls
// -----------------------------------------------------------------------------
//  The VECTORIZED variant of preprocessFusedClsDevice.
//
//  The math is unchanged -- same Pillow fixed-point chain, same coefficient
//  tables, same clip8, same /255 -- and the output is bit-identical.  Only the
//  memory access pattern and the amount of work per thread change:
//
//      scalar (preprocess_fused_yolov26_cls)  this file (vec)
//      -------------------------------------  ------------
//      1 thread = 1 output pixel              1 thread = 4 CONSECUTIVE output
//                                             pixels of one row (one quad)
//      3 scalar 4B stores per pixel           one 16B vector store per plane
//                                             (float4, streaming)
//      one integer division idx/outW per pixel one per quad (4x fewer)
//      the tap-row base pointer is recomputed the row pointer is loaded once and
//      by every pixel                          reused by all 4 lanes
//
//  ----------------------------------------------------------- why it is safe
//  1) Within one quad the 4 output pixels sit in the same row, hence share the
//     vertical window: the same (first, count) bounds and the same tap rows.
//     Only the horizontal windows differ, so pilHorizPixel is still called per
//     lane and each lane keeps its own integer accumulation.
//  2) The quad is contiguous inside the output plane, so the float4 store is
//     valid whenever the plane offset is 16B aligned.  That is guaranteed by
//     requiring outW % 4 == 0 (then oy*outW + ox0 is a multiple of 4); the plan
//     REJECTS any other output width instead of silently degrading, following
//     "config disagrees with the operator -> fail explicitly".
//  3) Every accumulator is int32 and every tap is added with an integer add, so
//     lane ordering and the compiler's reassociation cannot change the result --
//     unlike a float kernel, bit-exactness here is structural, not incidental.
//
//  WARNING: the output size is fixed (static ONNX); the geometry and the
//  definition of PreprocessPilClsConfig live in yolov26_cls_common.hpp, and the
//  scalar/vectorized shared arithmetic lives in pil_resample_detail.cuh.  This
//  header deliberately does NOT include the scalar operator header.
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/pil_resample.hpp"
#include "trt_alpha/kernels/ops/yolov26_cls_common.hpp"

#include <cuda_runtime.h>   // cudaStream_t / cudaError_t / cudaSuccess

#include <cstddef>
#include <memory>

namespace trt_alpha::kernels::ops {

// =============================================================================
//  CvPreprocessVecClsPlan -- interface identical to CvPreprocessClsPlan
// -----------------------------------------------------------------------------
//  Usage:
//      auto plan = CvPreprocessVecClsPlan::byDivisor(1280, 1280, 640, 640);
//      for (...) plan.launch(stream, batch, d_src, d_dst);
//
//  Constructing with an outW that is not a multiple of 4 records
//  cudaErrorInvalidValue (ok() == false) and never launches: the vectorized store
//  would be misaligned, which is a bug, not a slow path -- the scalar operator
//  should be used instead.
// =============================================================================
class CvPreprocessVecClsPlan
{
public:
    CvPreprocessVecClsPlan(int srcW, int srcH, int outW, int outH, float divisor = 255.0f);

    //! dst = src / divisor -- bit-exact with torch's uint8.float().div_(255.0)
    static CvPreprocessVecClsPlan byDivisor(int srcW, int srcH, int outW, int outH,
                                            float divisor = 255.0f);

    ~CvPreprocessVecClsPlan();

    CvPreprocessVecClsPlan(const CvPreprocessVecClsPlan&)            = delete;
    CvPreprocessVecClsPlan& operator=(const CvPreprocessVecClsPlan&) = delete;
    CvPreprocessVecClsPlan(CvPreprocessVecClsPlan&& other) noexcept;

    bool matches(int srcW, int srcH, int outW, int outH) const noexcept
    {
        return cfg_.srcW == srcW && cfg_.srcH == srcH
            && cfg_.outW == outW && cfg_.outH == outH;
    }

    const PreprocessPilClsConfig& info() const noexcept { return cfg_; }

    int outW() const noexcept { return cfg_.outW; }
    int outH() const noexcept { return cfg_.outH; }
    std::size_t outElems() const noexcept
    {
        return static_cast<std::size_t>(cfg_.outW) * cfg_.outH * 3;
    }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    //! Hot path: issues exactly one kernel; no allocation, no copy, no sync.
    void launch(cudaStream_t stream, int batch,
                const unsigned char* src, float* dst) const;

private:
    PreprocessPilClsConfig cfg_{};

    //! Only its fixed-point coefficient tables are borrowed -- the same tables the
    //! standalone CvPilResamplePlan is verified against, so the individual and
    //! fused paths cannot drift. CvPilResamplePlan is not assignable, hence the
    //! unique_ptr (the same shape as CvPreprocessVecPlan's CvResizePlan member).
    std::unique_ptr<CvPilResamplePlan> resample_;

    float divisor_ = 255.0f;

    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  preprocessFusedVecClsDevice -- one-shot convenience version
// =============================================================================
void preprocessFusedVecClsDevice(cudaStream_t stream, int batch,
                                 const unsigned char* src, float* dst,
                                 int srcW, int srcH, int outW, int outH,
                                 float divisor = 255.0f);

}  // namespace trt_alpha::kernels::ops
