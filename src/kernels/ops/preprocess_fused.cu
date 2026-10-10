// =============================================================================
//  trt_alpha :: kernels :: ops :: preprocess_fused
// -----------------------------------------------------------------------------
//  See the top of preprocess_fused.hpp for the full description. Key points:
//    * each output pixel needs only 2x2 source pixels -> the fixed-point chain can
//      be reproduced in place, with no intermediate buffer
//    * the coefficient tables come straight from CvResizePlan::coefTables()
//      (the set already verified to give max=0)
//    * padding pixels write a constant directly, without touching the source
// =============================================================================
#include "trt_alpha/kernels/ops/preprocess_fused.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <utility>

namespace trt_alpha::kernels::ops {

// -----------------------------------------------------------------------------
//  Fixed-size letterbox geometry -- the single source of truth.
//
//  Python (sry_preprocess_fused_dump.py / LetterBox(auto=False, center=True)):
//      r    = min(IMGSZ_H/h0, IMGSZ_W/w0)
//      nw   = round(w0*r);  nh = round(h0*r)            # round = half-to-even
//      dw   = (IMGSZ_W-nw)/2;  dh = (IMGSZ_H-nh)/2
//      left = int(round(dw-0.1)); right = int(round(dw+0.1))
//      top  = int(round(dh-0.1)); bottom= int(round(dh+0.1))
//  Reproduced statement by statement here: double arithmetic + nearbyint
//  (the default rounding mode is half-to-even).
// -----------------------------------------------------------------------------
LetterBoxInfo computeLetterBox(int srcW, int srcH, int outW, int outH)
{
    LetterBoxInfo info;

    const double rw = (double)outW / (double)srcW;
    const double rh = (double)outH / (double)srcH;
    const double r  = rw < rh ? rw : rh;

    info.srcW = srcW;
    info.srcH = srcH;
    info.outW = outW;
    info.outH = outH;
    info.ratio = r;
    info.newW = (int)std::nearbyint((double)srcW * r);
    info.newH = (int)std::nearbyint((double)srcH * r);

    const double dw = ((double)outW - (double)info.newW) / 2.0;
    const double dh = ((double)outH - (double)info.newH) / 2.0;
    info.padLeft   = (int)std::nearbyint(dw - 0.1);
    info.padRight  = (int)std::nearbyint(dw + 0.1);
    info.padTop    = (int)std::nearbyint(dh - 0.1);
    info.padBottom = (int)std::nearbyint(dh + 0.1);

    return info;
}

namespace {

constexpr int kCN = 3;   //!< fixed BGR 3 channels (this operator serves colour images only)

// =============================================================================
//  Fused kernel
// -----------------------------------------------------------------------------
//  Thread mapping: the x dimension flattens the outW*outH output pixels of one
//  image, the y dimension is the batch index. One thread processes the 3 channels
//  of that pixel (also taking care of BGR->RGB and of the CHW write position).
// =============================================================================
template <bool kAffine>
__global__ void preprocessFusedKernel(
    const unsigned char* __restrict__ src, int srcW, int srcH,
    float* __restrict__ dst, int outW, int outH,
    int newW, int newH, int padLeft, int padTop,
    const int*   __restrict__ xofs,
    const short* __restrict__ xalpha,
    const int*   __restrict__ yofs,
    const short* __restrict__ ybeta,
    float a, float b, float padNorm, int batch)
{
    const int p = blockDim.x * blockIdx.x + threadIdx.x;   // linear output pixel index
    const int n = blockIdx.y;                              // batch index
    if (p >= outW * outH || n >= batch)
    {
        return;
    }

    const int   oy    = p / outW;
    const int   ox    = p - oy * outW;
    const int   plane = outW * outH;
    float*      out   = dst + (size_t)n * (size_t)plane * kCN + p;

    // ---- padding: write the constant directly (skips the whole interpolation chain) ----
    const int ix = ox - padLeft;
    const int iy = oy - padTop;
    if (ix < 0 || ix >= newW || iy < 0 || iy >= newH)
    {
        out[0]           = padNorm;
        out[plane]       = padNorm;
        out[2 * plane]   = padNorm;
        return;
    }

    // ---- fixed-point interpolation (step-by-step bit operations of
    //      cv::resize(INTER_LINEAR, 8U), copied verbatim) ----
    int syA = yofs[iy];
    int syB = syA + 1;
    syA = syA < 0 ? 0 : (syA > srcH - 1 ? srcH - 1 : syA);
    syB = syB < 0 ? 0 : (syB > srcH - 1 ? srcH - 1 : syB);

    const int sx0 = xofs[ix];
    int       sx1 = sx0 + 1;
    if (sx1 > srcW - 1) sx1 = srcW - 1;

    const short a0 = xalpha[(size_t)ix * 2 + 0];
    const short a1 = xalpha[(size_t)ix * 2 + 1];
    const short b0 = ybeta [(size_t)iy * 2 + 0];
    const short b1 = ybeta [(size_t)iy * 2 + 1];

    const unsigned char* img = src + (size_t)n * (size_t)srcW * srcH * kCN;
    const unsigned char* rA0 = img + ((size_t)syA * srcW + sx0) * kCN;
    const unsigned char* rA1 = img + ((size_t)syA * srcW + sx1) * kCN;
    const unsigned char* rB0 = img + ((size_t)syB * srcW + sx0) * kCN;
    const unsigned char* rB1 = img + ((size_t)syB * srcW + sx1) * kCN;

#pragma unroll
    for (int c = 0; c < kCN; ++c)
    {
        const int sc = kCN - 1 - c;        // BGR -> RGB: output channel c takes source channel 2-c

        const int rowA = (int)rA0[sc] * a0 + (int)rA1[sc] * a1;
        const int rowB = (int)rB0[sc] * a0 + (int)rB1[sc] * a1;

        int val = (((b0 * (rowA >> 4)) >> 16)
                 + ((b1 * (rowB >> 4)) >> 16)
                 + 2) >> 2;
        if (val < 0)   val = 0;
        if (val > 255) val = 255;

        const float f = (float)val;
        out[(size_t)c * plane] = kAffine ? __fadd_rn(__fmul_rn(f, a), b)
                                         : __fdiv_rn(f, a);
    }
}

template <bool kAffine>
void launchOne(cudaStream_t stream, int batch,
               const unsigned char* src, float* dst,
               const LetterBoxInfo& info, float padNorm, float a, float b,
               const CvResizePlan<unsigned char, unsigned char>::CoefTables& t)
{
    const int pix = info.outW * info.outH;
    if (pix <= 0 || batch <= 0)
    {
        return;
    }
    const dim3 block(kBlock1D, 1);
    const dim3 grid((unsigned)((pix + kBlock1D - 1) / kBlock1D), (unsigned)batch);

    preprocessFusedKernel<kAffine><<<grid, block, 0, stream>>>(
        src, info.srcW, info.srcH, dst, info.outW, info.outH,
        info.newW, info.newH, info.padLeft, info.padTop,
        t.xofs, t.xalpha, t.yofs, t.ybeta,
        a, b, padNorm, batch);
}

}  // namespace

// =============================================================================
//  CvPreprocessPlan
// =============================================================================
CvPreprocessPlan::CvPreprocessPlan(int srcW, int srcH, int outW, int outH,
                                   double padValue, NormMode mode, float a, float b)
    : mode_(mode)
    , a_(a)
    , b_(b)
{
    if (srcW <= 0 || srcH <= 0 || outW <= 0 || outH <= 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }
    if (mode == NormMode::DivConst && a == 0.0f)
    {
        err_ = cudaErrorInvalidValue;   // division by zero
        return;
    }

    info_ = computeLetterBox(srcW, srcH, outW, outH);

    if (info_.newW < 1 || info_.newH < 1 ||
        info_.newW > outW || info_.newH > outH ||
        info_.padLeft < 0 || info_.padTop < 0 ||
        info_.padRight < 0 || info_.padBottom < 0)
    {
        err_ = cudaErrorInvalidValue;   // extreme aspect ratio (one side < 1 pixel after scaling)
        return;
    }

    // Fixed-point coefficients: srcW x srcH -> newW x newH, the same set as
    // cv::resize (already verified to give max=0)
    resize_.reset(new CvResizePlan<unsigned char, unsigned char>(
        srcW, srcH, info_.newW, info_.newH));
    if (!resize_->ok())
    {
        err_ = resize_->lastError();
        return;
    }

    // Padding value: the reference chain "fills padValue in the u8 domain first,
    // then divides by 255", so round to u8 first
    const float fpad = (float)detail::saturateCastU8(padValue);
    padNorm_ = (mode == NormMode::Affine) ? (fpad * a + b)
                                          : (fpad / a);
    err_ = cudaSuccess;
}

CvPreprocessPlan CvPreprocessPlan::byDivisor(int srcW, int srcH, int outW, int outH,
                                             double padValue, float divisor)
{
    return CvPreprocessPlan(srcW, srcH, outW, outH, padValue, NormMode::DivConst, divisor, 0.0f);
}

CvPreprocessPlan CvPreprocessPlan::byAffine(int srcW, int srcH, int outW, int outH,
                                            double padValue, float alpha, float beta)
{
    return CvPreprocessPlan(srcW, srcH, outW, outH, padValue, NormMode::Affine, alpha, beta);
}

CvPreprocessPlan::~CvPreprocessPlan() = default;

CvPreprocessPlan::CvPreprocessPlan(CvPreprocessPlan&& other) noexcept = default;

void CvPreprocessPlan::launch(cudaStream_t stream, int batch,
                              const unsigned char* src, float* dst) const
{
    if (err_ != cudaSuccess || !resize_)
    {
        return;
    }
    const auto t = resize_->coefTables();
    if (mode_ == NormMode::Affine)
    {
        launchOne<true>(stream, batch, src, dst, info_, padNorm_, a_, b_, t);
    }
    else
    {
        launchOne<false>(stream, batch, src, dst, info_, padNorm_, a_, b_, t);
    }
    err_ = cudaGetLastError();
}

// =============================================================================
//  preprocessFusedDevice
// =============================================================================
void preprocessFusedDevice(cudaStream_t stream, int batch,
                           const unsigned char* src, float* dst,
                           int srcW, int srcH, int outW, int outH,
                           double padValue, NormMode mode, float a, float b)
{
    const CvPreprocessPlan plan(srcW, srcH, outW, outH, padValue, mode, a, b);
    if (!plan.ok())
    {
        return;
    }
    plan.launch(stream, batch, src, dst);
}

}  // namespace trt_alpha::kernels::ops
