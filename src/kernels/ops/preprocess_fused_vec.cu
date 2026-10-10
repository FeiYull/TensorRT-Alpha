// =============================================================================
//  trt_alpha :: kernels :: ops :: preprocess_fused_vec
// -----------------------------------------------------------------------------
//  Implementation described at the top of preprocess_fused_vec.hpp. Key points:
//    * the math is IDENTICAL to the scalar version; only the memory access changes
//      (1 thread = 4 pixels + float4 vector stores)
//    * the fixed-point coefficient tables still come from
//      CvResizePlan::coefTables() (the verified max=0 set)
//    * padding pixels still write a constant directly, without touching the source
//    * the letterbox geometry comes from the shared computeLetterBox() -- there is
//      no longer a duplicated copy to keep in sync
// =============================================================================
#include "trt_alpha/kernels/ops/preprocess_fused_vec.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <utility>

namespace trt_alpha::kernels::ops {
namespace {

constexpr int kCN      = 3;   //!< fixed BGR 3 channels
constexpr int kVec     = 4;   //!< 4 consecutive output pixels per thread (one float4)
constexpr int kVecMask = kVec - 1;

// =============================================================================
//  Vectorized fused kernel
// -----------------------------------------------------------------------------
//  Thread mapping: the x dimension covers the quads (4 consecutive output pixels)
//  of one image, the y dimension is the batch index.
//  Per quad:
//      1 integer division to get (oy, ox) -> the other 3 pixels advance by
//        increments (no further division)
//      per pixel, the fixed-point interpolation of 3 channels (including the
//        BGR->RGB channel pick)
//      one float4 store per plane (3 vector stores for the 3 channels)
// =============================================================================
template <bool kAffine>
__global__ void preprocessFusedVecKernel(
    const unsigned char* __restrict__ src, int srcW, int srcH,
    float* __restrict__ dst, int outW, int outH,
    int newW, int newH, int padLeft, int padTop,
    const int*   __restrict__ xofs,
    const short* __restrict__ xalpha,
    const int*   __restrict__ yofs,
    const short* __restrict__ ybeta,
    float a, float b, float padNorm, int batch)
{
    const int plane = outW * outH;
    const int n     = blockIdx.y;
    const int p0    = ((int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x) * kVec;
    if (p0 >= plane || n >= batch)
    {
        return;
    }

    const int  nValid   = (plane - p0) < kVec ? (plane - p0) : kVec;
    const bool vecStore = ((plane & kVecMask) == 0);   // plane divisible by 4 -> 16B aligned

    const unsigned char* img = src + (size_t)n * (size_t)srcW * srcH * kCN;

    float vals[kCN][kVec];

    // ---- one integer division only; the other 3 pixels advance by increments ----
    int oy = p0 / outW;
    int ox = p0 - oy * outW;

#pragma unroll
    for (int k = 0; k < kVec; ++k)
    {
        if (k >= nValid) break;

        const int ix = ox - padLeft;
        const int iy = oy - padTop;

        if (ix < 0 || ix >= newW || iy < 0 || iy >= newH)
        {
            // padding: write the constant directly, skipping the whole interpolation chain
            vals[0][k] = padNorm;
            vals[1][k] = padNorm;
            vals[2][k] = padNorm;
        }
        else
        {
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
                vals[c][k] = kAffine ? __fadd_rn(__fmul_rn(f, a), b)
                                     : __fdiv_rn(f, a);
            }
        }

        if (++ox == outW) { ox = 0; ++oy; }
    }

    // ---- store: one 16B vector store per plane (falls back to scalar when the
    //      plane is not divisible by 4) ----
    float* out = dst + (size_t)n * (size_t)plane * kCN + p0;

#pragma unroll
    for (int c = 0; c < kCN; ++c)
    {
        float* q = out + (size_t)c * plane;
        if (vecStore && nValid == kVec)
        {
            __stcs(reinterpret_cast<float4*>(q),
                   make_float4(vals[c][0], vals[c][1], vals[c][2], vals[c][3]));
        }
        else
        {
            for (int k = 0; k < nValid; ++k) q[k] = vals[c][k];
        }
    }
}

template <bool kAffine>
void launchOneVec(cudaStream_t stream, int batch,
                  const unsigned char* src, float* dst,
                  const LetterBoxInfo& info, float padNorm, float a, float b,
                  const CvResizePlan<unsigned char, unsigned char>::CoefTables& t)
{
    const int pix = info.outW * info.outH;
    if (pix <= 0 || batch <= 0)
    {
        return;
    }
    const int  quads = (pix + kVec - 1) / kVec;
    const dim3 block(kBlock1D, 1);
    const dim3 grid((unsigned)((quads + kBlock1D - 1) / kBlock1D), (unsigned)batch);

    preprocessFusedVecKernel<kAffine><<<grid, block, 0, stream>>>(
        src, info.srcW, info.srcH, dst, info.outW, info.outH,
        info.newW, info.newH, info.padLeft, info.padTop,
        t.xofs, t.xalpha, t.yofs, t.ybeta,
        a, b, padNorm, batch);
}

}  // namespace

// =============================================================================
//  CvPreprocessVecPlan
// =============================================================================
CvPreprocessVecPlan::CvPreprocessVecPlan(int srcW, int srcH, int outW, int outH,
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
        err_ = cudaErrorInvalidValue;
        return;
    }

    // Shared with the scalar version -- one single geometry implementation.
    info_ = computeLetterBox(srcW, srcH, outW, outH);

    if (info_.newW < 1 || info_.newH < 1 ||
        info_.newW > outW || info_.newH > outH ||
        info_.padLeft < 0 || info_.padTop < 0 ||
        info_.padRight < 0 || info_.padBottom < 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    resize_.reset(new CvResizePlan<unsigned char, unsigned char>(
        srcW, srcH, info_.newW, info_.newH));
    if (!resize_->ok())
    {
        err_ = resize_->lastError();
        return;
    }

    const float fpad = (float)detail::saturateCastU8(padValue);
    padNorm_ = (mode == NormMode::Affine) ? (fpad * a + b) : (fpad / a);
    err_ = cudaSuccess;
}

CvPreprocessVecPlan CvPreprocessVecPlan::byDivisor(int srcW, int srcH, int outW, int outH,
                                                   double padValue, float divisor)
{
    return CvPreprocessVecPlan(srcW, srcH, outW, outH, padValue, NormMode::DivConst, divisor, 0.0f);
}

CvPreprocessVecPlan CvPreprocessVecPlan::byAffine(int srcW, int srcH, int outW, int outH,
                                                  double padValue, float alpha, float beta)
{
    return CvPreprocessVecPlan(srcW, srcH, outW, outH, padValue, NormMode::Affine, alpha, beta);
}

CvPreprocessVecPlan::~CvPreprocessVecPlan() = default;

CvPreprocessVecPlan::CvPreprocessVecPlan(CvPreprocessVecPlan&& other) noexcept = default;

void CvPreprocessVecPlan::launch(cudaStream_t stream, int batch,
                                 const unsigned char* src, float* dst) const
{
    if (err_ != cudaSuccess || !resize_)
    {
        return;
    }
    const auto t = resize_->coefTables();
    if (mode_ == NormMode::Affine)
    {
        launchOneVec<true>(stream, batch, src, dst, info_, padNorm_, a_, b_, t);
    }
    else
    {
        launchOneVec<false>(stream, batch, src, dst, info_, padNorm_, a_, b_, t);
    }
    err_ = cudaGetLastError();
}

// =============================================================================
//  preprocessFusedVecDevice
// =============================================================================
void preprocessFusedVecDevice(cudaStream_t stream, int batch,
                              const unsigned char* src, float* dst,
                              int srcW, int srcH, int outW, int outH,
                              double padValue, NormMode mode, float a, float b)
{
    const CvPreprocessVecPlan plan(srcW, srcH, outW, outH, padValue, mode, a, b);
    if (!plan.ok())
    {
        return;
    }
    plan.launch(stream, batch, src, dst);
}

}  // namespace trt_alpha::kernels::ops
