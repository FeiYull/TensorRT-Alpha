// =============================================================================
//  trt_alpha :: kernels :: ops :: preprocess_fused_vec_yolov26_cls  (implementation)
// -----------------------------------------------------------------------------
//  Implementation described at the top of preprocess_fused_vec_yolov26_cls.hpp.
//  Key points:
//    * the math is IDENTICAL to the scalar version -- same CvPilResamplePlan
//      coefficient tables, same clip8, same /255 -- only the memory access
//      pattern and the work per thread change
//    * 1 thread = 4 CONSECUTIVE output pixels of one row (one quad): the quad
//      shares the vertical window (one (first, count) pair and one row pointer
//      per tap) but every lane still runs its own pilHorizPixel, so each lane
//      keeps its own int32 accumulation
//    * every accumulator is int32 and every tap is an integer add, therefore
//      lane ordering and the compiler's reassociation cannot change the result:
//      bit-exactness against the scalar operator is structural, not incidental
//    * there is NO scalar tail and NO fallback here: the geometry helper REJECTS
//      outW % 4 != 0 by recording cudaErrorInvalidValue (a misaligned float4
//      store is a bug, not a slow path), following "the operator cannot honour
//      the request -> fail explicitly"
// =============================================================================
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_cls.hpp"

#include "pil_resample_detail.cuh"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace trt_alpha::kernels::ops {
namespace {

constexpr int kCN      = 3;   //!< fixed BGR 3 channels
constexpr int kVec     = 4;   //!< 4 consecutive output pixels per thread (one float4)
constexpr int kVecMask = kVec - 1;

// =============================================================================
//  Vectorized fused kernel (resize + center crop + BGR->RGB + HWC->CHW + /divisor)
// -----------------------------------------------------------------------------
//  Thread mapping: the x dimension covers the QUADS of one image, the y dimension
//  is the batch index.
//  Per quad:
//      1 integer division to get (oy, ox0)            -> the other 3 pixels only
//                                                        need +1/+2/+3
//      the vertical window and the tap row pointer are loaded ONCE per tap and
//        reused by all 4 lanes
//      the horizontal window is per lane (pilHorizPixel), so all 4 lanes have
//        independent integer accumulators
//      one float4 store per plane (3 vector stores for the 3 channels)
//
//  The horizontal pass is row preserving, so resized row r is built from source
//  row r -- no temporary row offset is involved (same as the scalar operator).
// =============================================================================
__global__ void preprocessFusedVecClsKernel(const unsigned char* __restrict__ src,
                                            float* __restrict__ dst,
                                            int srcW, int srcH,
                                            int outW, int outH,
                                            int cropLeft, int cropTop,
                                            int needH, int needV,
                                            const PilBound* __restrict__ bH,
                                            const int* __restrict__ kH, int ksizeH,
                                            const PilBound* __restrict__ bV,
                                            const int* __restrict__ kV, int ksizeV,
                                            float divisor)
{
    // outW is a multiple of 4 (enforced by the plan), so the two shifts below
    // are exact and the quad stays inside a single row.
    const int quadsPerRow = outW >> 2;
    const int nq          = (outW * outH) >> 2;

    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    if (idx >= nq)
    {
        return;
    }

    const int oy  = idx / quadsPerRow;              // one division per quad
    const int ox0 = (idx - oy * quadsPerRow) << 2;  // first output column of the quad

    const unsigned char* im =
        src + static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(srcW) * srcH * kCN;

    const int ry  = oy + cropTop;                   // row    in the resized image
    const int rx0 = ox0 + cropLeft;                 // column in the resized image

    unsigned char r0[kVec], r1[kVec], r2[kVec];     // B, G, R of the 4 finished pixels

    if (needV)
    {
        // _ImagingResampleVertical_8bpc, bands == 3 (Resample.c:446).
        const PilBound bv = bV[ry];
        const int*     kv = kV + static_cast<std::size_t>(ry) * ksizeV;

        int s0[kVec], s1[kVec], s2[kVec];
#pragma unroll
        for (int l = 0; l < kVec; ++l)
        {
            // Pillow adds half of the fixed-point LSB so clip8's truncating shift
            // rounds to nearest (see pil_resample_detail.cuh).
            s0[l] = kPilRoundHalf;
            s1[l] = kPilRoundHalf;
            s2[l] = kPilRoundHalf;
        }

        for (int t = 0; t < bv.count; ++t)
        {
            const unsigned char* row =
                im + static_cast<std::size_t>(bv.first + t) * static_cast<std::size_t>(srcW) * kCN;
            const int kt = kv[t];                   // hoisted: shared by all 4 lanes
#pragma unroll
            for (int l = 0; l < kVec; ++l)
            {
                const uchar3 h = detail::pilHorizPixel(row, rx0 + l, needH, bH, kH, ksizeH);
                s0[l] += h.x * kt;
                s1[l] += h.y * kt;
                s2[l] += h.z * kt;
            }
        }

#pragma unroll
        for (int l = 0; l < kVec; ++l)
        {
            r0[l] = detail::pilClip8(s0[l]);
            r1[l] = detail::pilClip8(s1[l]);
            r2[l] = detail::pilClip8(s2[l]);
        }
    }
    else
    {
        // The vertical pass is skipped: the resized image IS the horizontally
        // resampled one, so the pixels are taken as is (still clipped there).
        const unsigned char* row = im + static_cast<std::size_t>(ry) * static_cast<std::size_t>(srcW) * kCN;
#pragma unroll
        for (int l = 0; l < kVec; ++l)
        {
            const uchar3 h = detail::pilHorizPixel(row, rx0 + l, needH, bH, kH, ksizeH);
            r0[l] = h.x;
            r1[l] = h.y;
            r2[l] = h.z;
        }
    }

    // BGR -> RGB, HWC -> CHW, / divisor.  Resize and crop act per channel, so
    // permuting the channels after the geometry is the same as before it.
    //
    // `idx` counts QUADS, so the first pixel of the quad is at pixel index
    // idx * kVec -- using idx directly would both write the wrong location and
    // break the 16B alignment.  dst comes from cudaMalloc and the offset is a
    // multiple of kVec floats (== 16B), so the float4 store is always aligned;
    // that is precisely why the plan rejects outW % 4 != 0.
    const std::size_t pl  = static_cast<std::size_t>(outW) * outH;
    float* const      out = dst + static_cast<std::size_t>(blockIdx.y) * 3 * pl
                          + static_cast<std::size_t>(idx) * kVec;

    __stcs(reinterpret_cast<float4*>(out),
           make_float4(static_cast<float>(r2[0]) / divisor,
                       static_cast<float>(r2[1]) / divisor,
                       static_cast<float>(r2[2]) / divisor,
                       static_cast<float>(r2[3]) / divisor));            // R plane
    __stcs(reinterpret_cast<float4*>(out + pl),
           make_float4(static_cast<float>(r1[0]) / divisor,
                       static_cast<float>(r1[1]) / divisor,
                       static_cast<float>(r1[2]) / divisor,
                       static_cast<float>(r1[3]) / divisor));            // G plane
    __stcs(reinterpret_cast<float4*>(out + 2 * pl),
           make_float4(static_cast<float>(r0[0]) / divisor,
                       static_cast<float>(r0[1]) / divisor,
                       static_cast<float>(r0[2]) / divisor,
                       static_cast<float>(r0[3]) / divisor));            // B plane
}

}  // namespace

// =============================================================================
//  CvPreprocessVecClsPlan
// =============================================================================
CvPreprocessVecClsPlan::CvPreprocessVecClsPlan(int srcW, int srcH, int outW, int outH,
                                              float divisor)
    : divisor_(divisor)
{
    // Geometry comes from the same helper as the scalar fused operator.
    cfg_ = computePilClsGeometry(srcW, srcH, outW, outH);
    if (!cfg_.valid())
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    // The vectorized store requires the row stride to be a multiple of 4, hence
    // outW % 4 == 0.  Rejecting here is deliberate: silently falling back to a
    // scalar tail (or to the scalar operator) would hide a caller mistake.
    if ((cfg_.outW & kVecMask) != 0)
    {
        TRT_LOG_ERROR("yolov26_cls vectorized preprocess: output width "
                      + std::to_string(cfg_.outW)
                      + " is not a multiple of 4; the float4 store would be"
                        " misaligned -- use the scalar operator instead");
        err_ = cudaErrorInvalidValue;
        return;
    }

    // The individual operator owns the Pillow geometry AND the fixed-point
    // coefficient tables; the fused kernel borrows both, so the step-by-step
    // chain and the fused path are guaranteed to use identical numbers.
    resample_.reset(new CvPilResamplePlan(srcW, srcH, cfg_.newW, cfg_.newH));
    if (!resample_->ok())
    {
        err_ = resample_->lastError();
        return;
    }

    cfg_.needH  = resample_->needH() ? 1 : 0;
    cfg_.needV  = resample_->needV() ? 1 : 0;
    cfg_.ksizeH = resample_->ksizeH();
    cfg_.ksizeV = resample_->ksizeV();

    err_ = cudaSuccess;
}

CvPreprocessVecClsPlan CvPreprocessVecClsPlan::byDivisor(int srcW, int srcH, int outW, int outH,
                                                        float divisor)
{
    return CvPreprocessVecClsPlan(srcW, srcH, outW, outH, divisor);
}

CvPreprocessVecClsPlan::CvPreprocessVecClsPlan(CvPreprocessVecClsPlan&& other) noexcept = default;

CvPreprocessVecClsPlan::~CvPreprocessVecClsPlan() = default;

void CvPreprocessVecClsPlan::launch(cudaStream_t stream, int batch,
                                    const unsigned char* src, float* dst) const
{
    if (err_ != cudaSuccess || !resample_ || batch <= 0 || src == nullptr || dst == nullptr)
    {
        return;
    }

    const auto t = resample_->coefTables();

    const int  nq    = (cfg_.outW * cfg_.outH) >> 2;
    const dim3 block(kBlock1D, 1);
    const dim3 grid(static_cast<unsigned>((nq + kBlock1D - 1) / kBlock1D),
                    static_cast<unsigned>(batch));

    preprocessFusedVecClsKernel<<<grid, block, 0, stream>>>(
        src, dst,
        cfg_.srcW, cfg_.srcH,
        cfg_.outW, cfg_.outH,
        cfg_.cropLeft, cfg_.cropTop,
        cfg_.needH, cfg_.needV,
        t.boundsH, t.coefH, t.ksizeH,
        t.boundsV, t.coefV, t.ksizeV,
        divisor_);

    err_ = cudaGetLastError();
}

// =============================================================================
//  preprocessFusedVecClsDevice
// =============================================================================
void preprocessFusedVecClsDevice(cudaStream_t stream, int batch,
                                 const unsigned char* src, float* dst,
                                 int srcW, int srcH, int outW, int outH,
                                 float divisor)
{
    CvPreprocessVecClsPlan plan(srcW, srcH, outW, outH, divisor);
    if (!plan.ok())
    {
        TRT_LOG_ERROR("yolov26_cls vectorized preprocess: one-shot plan is unusable");
        return;
    }
    plan.launch(stream, batch, src, dst);
}

}  // namespace trt_alpha::kernels::ops
