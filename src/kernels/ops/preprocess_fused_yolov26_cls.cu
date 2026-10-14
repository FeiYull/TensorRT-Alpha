// =============================================================================
//  trt_alpha :: kernels :: ops :: preprocess_fused_yolov26_cls  (implementation)
// -----------------------------------------------------------------------------
//  Host side : the geometry (Resize + CenterCrop) composed from the two
//              individual operators' geometry helpers, plus a CvPilResamplePlan
//              whose fixed-point coefficient tables are borrowed.
//  Device side: one kernel doing resize + crop + BGR->RGB + HWC->CHW + /255.
//
//  The whole chain is the torchvision/pillow one; see yolov26_cls_common.hpp and
//  pil_resample.hpp for the reference line numbers it is ported from
//  (torchvision 0.29.0 / Pillow 12.3.0 / torch 2.14.0 on this machine).
// =============================================================================
#include "trt_alpha/kernels/ops/preprocess_fused_yolov26_cls.hpp"

#include "pil_resample_detail.cuh"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/kernels/ops/center_crop.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace trt_alpha::kernels::ops {

// =============================================================================
//  computePilClsGeometry
// =============================================================================
PreprocessPilClsConfig computePilClsGeometry(int srcW, int srcH, int outW, int outH)
{
    PreprocessPilClsConfig cfg;
    cfg.srcW = srcW;
    cfg.srcH = srcH;
    cfg.outW = outW;
    cfg.outH = outH;

    if (srcW <= 0 || srcH <= 0 || outW <= 0 || outH <= 0)
    {
        return cfg;      // valid() is already false
    }

    // ultralytics classify_transforms(): a square target uses the scalar
    // shortest-edge mode, a non-square one resizes to that exact (h, w).
    if (outW == outH)
    {
        // Reuses the individual operator's geometry helper -- one implementation.
        computePilResizeShortestEdge(srcW, srcH, outW, cfg.newW, cfg.newH);
    }
    else
    {
        cfg.newW = outW;
        cfg.newH = outH;
    }

    // torchvision functional.center_crop, i.e. int(round((image - crop) / 2.0))
    // with Python's half-to-even rounding -- again the individual operator's
    // helper, so the fused and the step-by-step chain cannot disagree.
    //
    // When the resize came out smaller than the crop, CenterCrop would zero-pad;
    // that path is not implemented, center_crop leaves the offsets at 0 and the
    // resulting config fails valid(), which the plan turns into an error.
    const CenterCropInfo cc = computeCenterCrop(cfg.newW, cfg.newH, outW, outH);
    cfg.cropLeft = cc.left;
    cfg.cropTop  = cc.top;

    // Pillow skips a pass whose output extent equals its input extent.
    cfg.needH = (cfg.newW != srcW) ? 1 : 0;
    cfg.needV = (cfg.newH != srcH) ? 1 : 0;

    return cfg;
}

// =============================================================================
//  device kernel
// =============================================================================
namespace {

//! One thread = one output pixel (all 3 channels together).
//!
//! Writes: the ox of 32 threads within a warp are consecutive, so each of the
//! three plane stores is one coalesced transaction.
//! Reads : each tap row is read from L1/L2 -- neighbouring output pixels share
//! their source window whenever the image is downscaled, and the quad variants
//! in the vectorized operator additionally share the row pointer itself.
__global__ void preprocessFusedClsKernel(const unsigned char* __restrict__ src,
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
    const int plane = outW * outH;
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plane)
    {
        return;
    }

    const int ox = idx % outW;
    const int oy = idx / outW;

    const unsigned char* im = src + static_cast<std::size_t>(blockIdx.y) * srcW * srcH * 3;
    const int rx = ox + cropLeft;        // column in the resized image
    const int ry = oy + cropTop;         // row    in the resized image

    unsigned char r0, r1, r2;            // B, G, R of the finished pixel
    if (needV)
    {
        // _ImagingResampleVertical_8bpc, bands == 3 (Resample.c:446).
        // The horizontal pass is row preserving, so resized row r is built from
        // source row r -- no temporary row offset is involved.
        const PilBound bv = bV[ry];
        const int*     kv = kV + static_cast<std::size_t>(ry) * ksizeV;

        int s0 = kPilRoundHalf, s1 = kPilRoundHalf, s2 = kPilRoundHalf;
        for (int t = 0; t < bv.count; ++t)
        {
            const unsigned char* row =
                im + static_cast<std::size_t>(bv.first + t) * srcW * 3;
            const uchar3 h  = detail::pilHorizPixel(row, rx, needH, bH, kH, ksizeH);
            const int    kt = kv[t];
            s0 += h.x * kt;
            s1 += h.y * kt;
            s2 += h.z * kt;
        }
        r0 = detail::pilClip8(s0);
        r1 = detail::pilClip8(s1);
        r2 = detail::pilClip8(s2);
    }
    else
    {
        // The vertical pass is skipped: the resized image IS the horizontally
        // resampled one, so the pixel is taken as is (still possibly filtered
        // horizontally, still clipped there).
        const unsigned char* row = im + static_cast<std::size_t>(ry) * srcW * 3;
        const uchar3         h   = detail::pilHorizPixel(row, rx, needH, bH, kH, ksizeH);
        r0 = h.x;
        r1 = h.y;
        r2 = h.z;
    }

    // BGR -> RGB, HWC -> CHW, / 255.  Resize and crop act per channel, so
    // permuting the channels after the geometry is the same as before it.
    const std::size_t pl  = static_cast<std::size_t>(plane);
    float* const      out = dst + static_cast<std::size_t>(blockIdx.y) * 3 * pl + idx;
    out[0]      = static_cast<float>(r2) / divisor;   // R
    out[pl]     = static_cast<float>(r1) / divisor;   // G
    out[2 * pl] = static_cast<float>(r0) / divisor;   // B
}

}  // namespace

// =============================================================================
//  CvPreprocessClsPlan
// =============================================================================
CvPreprocessClsPlan::CvPreprocessClsPlan(int srcW, int srcH, int outW, int outH, float divisor)
    : divisor_(divisor)
{
    cfg_ = computePilClsGeometry(srcW, srcH, outW, outH);
    if (!cfg_.valid())
    {
        // Non-positive sizes, or a resize smaller than the crop (which would
        // need CenterCrop's zero padding).
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

CvPreprocessClsPlan CvPreprocessClsPlan::byDivisor(int srcW, int srcH, int outW, int outH,
                                                  float divisor)
{
    return CvPreprocessClsPlan(srcW, srcH, outW, outH, divisor);
}

CvPreprocessClsPlan::CvPreprocessClsPlan(CvPreprocessClsPlan&& other) noexcept = default;

CvPreprocessClsPlan::~CvPreprocessClsPlan() = default;

void CvPreprocessClsPlan::launch(cudaStream_t stream, int batch,
                                 const unsigned char* src, float* dst) const
{
    if (err_ != cudaSuccess || !resample_ || batch <= 0 || src == nullptr || dst == nullptr)
    {
        return;
    }

    const auto t = resample_->coefTables();

    const int  plane = cfg_.outW * cfg_.outH;
    const int  block = 256;
    const dim3 grid(static_cast<unsigned>((plane + block - 1) / block),
                    static_cast<unsigned>(batch));

    preprocessFusedClsKernel<<<grid, block, 0, stream>>>(
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
//  preprocessFusedClsDevice
// =============================================================================
void preprocessFusedClsDevice(cudaStream_t stream, int batch,
                              const unsigned char* src, float* dst,
                              int srcW, int srcH, int outW, int outH,
                              float divisor)
{
    CvPreprocessClsPlan plan(srcW, srcH, outW, outH, divisor);
    if (!plan.ok())
    {
        TRT_LOG_ERROR("yolov26_cls preprocess: one-shot plan is unusable");
        return;
    }
    plan.launch(stream, batch, src, dst);
}

}  // namespace trt_alpha::kernels::ops
