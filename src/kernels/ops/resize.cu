// =============================================================================
//  trt_alpha :: kernels :: ops :: resize
// -----------------------------------------------------------------------------
//  CUDA-C reproduction of OpenCV 4.8.1's cv::resize(INTER_LINEAR).
//
//  OpenCV reference implementation (sources/modules/imgproc/src/resize.cpp):
//      hal::resize()                                  -- builds xofs/xalpha/yofs/ybeta
//      resizeGeneric_<HResizeLinear, VResizeLinear>   -- horizontal + vertical passes
//
//  Two completely different numeric paths (resize.cpp:3926: bool fixpt = depth == CV_8U):
//
//    [8U -> 8U]  fixed-point chain (linear_tab[CV_8U])
//        horizontal: rowA = S[sx]*a0 + S[sx+cn]*a1        (int; a0/a1 are short, scaled by 2048)
//        vertical:   v = ((b0*(rowA>>4))>>16 + (b1*(rowB>>4))>>16 + 2) >> 2
//        coefficients: cbuf = (1-fx, fx), ialpha = saturate_cast<short>(cbuf * 2048)
//              saturate_cast<short>(float) = cvRound + clamp; cvRound is round-half-to-even
//
//    [32F -> 32F] two float passes (linear_tab[CV_32F], ONE = 1)
//        horizontal: t = S[sx]*a0 + S[sx+cn]*a1           (float; a0/a1 are float, no fixed point)
//        vertical:   dst = S0[x]*b0 + S1[x]*b1
//        coefficients: alpha is simply the float (1-fx, fx); no fixed-point scaling,
//                      no rounding
//
//  Coordinate sequence (resize.cpp:3819 / 3955 / 4072) -- must be copied bit for bit:
//      inv_scale_x = (double)dstW / srcW;     scale_x = 1. / inv_scale_x;
//      fx = (float)((dx + 0.5) * scale_x - 0.5);  sx = cvFloor(fx);  fx -= sx;
//      edge clamping (when ksize2 == 1):
//          sx < 0            -> fx = 0, sx = 0
//          sx >= srcW - 1    -> fx = 0, sx = srcW - 1
//
//  Multi-channel: OpenCV expands each row to dstW*cn columns and replicates the
//  ksize weights of a pixel across cn channels (resize.cpp:3980-4005). That is
//  equivalent to: the same weight set interpolates each of the cn channels
//  independently across two neighbouring source pixels. This implementation
//  follows that equivalent form channel by channel and is bit-exact with OpenCV.
// =============================================================================
#include "trt_alpha/kernels/ops/resize.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace trt_alpha::kernels::ops {
namespace {

// -----------------------------------------------------------------------------
// Fixed-point constants and rounding (equivalent to OpenCV's
// INTER_RESIZE_COEF_SCALE / cvRound)
// -----------------------------------------------------------------------------
constexpr int kCoefScale = 1 << 11;   // 2048

inline int roundHalfToEven(float v) { return (int)rintf(v); }

inline short saturateCastShort(float v)
{
    int iv = roundHalfToEven(v);
    if (iv < -32768) iv = -32768;
    if (iv >  32767) iv =  32767;
    return (short)iv;
}

//! Turn a "normalized weight" into the coefficient type used by that OpenCV path.
template <typename CoefT> struct CoefMake;
template <> struct CoefMake<short>
{
    static short make(float w) { return saturateCastShort(w * (float)kCoefScale); }
};
template <> struct CoefMake<float>
{
    static float make(float w) { return w; }
};

// -----------------------------------------------------------------------------
// Precompute interpolation coefficients (depends on geometry only, not on cn/batch)
//
// The coordinate sequences of the two paths are DELIBERATELY different, because of
// one measured fact:
//
//   * 8U + INTER_LINEAR: OpenCV explicitly excludes it from IPP
//       (resize.cpp:3598 `ippDataType == ipp8u && ippInter == ippLinear` -> return false),
//       so it runs OpenCV's own fixed-point code. That code first computes
//       `fx = (float)((dx+0.5)*scale_x - 0.5)`, truncating fx to float (at an fx
//       magnitude of ~500 the fractional part only has ~6e-5 resolution), then does
//       `fx -= sx` and finally saturate_cast<short>. This truncation must be
//       reproduced bit for bit, otherwise the short coefficients change and
//       bit-exactness is lost.
//
//   * 32F + INTER_LINEAR: OpenCV hands it to IPP by default (the exclusion above
//       does not cover it), and IPP computes weights at sub-pixel precision without
//       that float truncation. Measured: the IPP result differs from the exact
//       double-precision value by ~3 ulp, while OpenCV's own C++ 32F path is off by
//       as much as ~0.007 (value range 0..255). To match cv::resize's default
//       runtime output, the 32F path computes the fractional part in double.
//
//   Measured (700x450 -> 1000x700, value range 0..255):
//       |CUDA - cv::resize(default, IPP)| = 3e-5
//       |CUDA - cv::resize(setUseIPP(false))| = 0.007   (i.e. the OpenCV C++ source path)
// -----------------------------------------------------------------------------
template <typename CoefT>
void buildCoeffs(int srcW, int srcH, int dstW, int dstH,
                 std::vector<int>&     xofs,
                 std::vector<CoefT>&   xalpha,
                 std::vector<int>&     yofs,
                 std::vector<CoefT>&   ybeta)
{
    // Exactly like OpenCV: compute inv_scale first, then take the reciprocal
    // (rather than dividing directly).
    const double inv_scale_x = (double)dstW / (double)srcW;
    const double inv_scale_y = (double)dstH / (double)srcH;
    const double scale_x     = 1.0 / inv_scale_x;
    const double scale_y     = 1.0 / inv_scale_y;

    // 8U: reproduce OpenCV C++'s float truncation; 32F: sub-pixel exact (match IPP)
    constexpr bool kReplicaCpp = std::is_same<CoefT, short>::value;

    xofs.resize((size_t)dstW);
    xalpha.resize((size_t)dstW * 2);
    yofs.resize((size_t)dstH);
    ybeta.resize((size_t)dstH * 2);

    // ---- x direction: clamping is done here (same as OpenCV's precomputation) ----
    for (int dx = 0; dx < dstW; ++dx)
    {
        const double fxd = (double)(dx + 0.5) * scale_x - 0.5;
        float fx;
        int   sx;
        if (kReplicaCpp)
        {
            fx = (float)fxd;                       // OpenCV: fx is truncated to float first
            sx = (int)std::floor(fx);
            fx -= (float)sx;
        }
        else
        {
            sx = (int)std::floor(fxd);             // sub-pixel exact: fraction stays in double
            fx = (float)(fxd - (double)sx);
        }

        if (sx < 0)         { fx = 0.f; sx = 0; }
        if (sx >= srcW - 1) { fx = 0.f; sx = srcW - 1; }

        xofs[dx]                   = sx;
        xalpha[(size_t)dx * 2 + 0] = CoefMake<CoefT>::make(1.f - fx);
        xalpha[(size_t)dx * 2 + 1] = CoefMake<CoefT>::make(fx);
    }

    // ---- y direction: no clamping (sy may be negative); the kernel clips to [0, srcH-1] ----
    for (int dy = 0; dy < dstH; ++dy)
    {
        const double fyd = (double)(dy + 0.5) * scale_y - 0.5;
        float fy;
        int   sy;
        if (kReplicaCpp)
        {
            fy = (float)fyd;
            sy = (int)std::floor(fy);
            fy -= (float)sy;
        }
        else
        {
            sy = (int)std::floor(fyd);
            fy = (float)(fyd - (double)sy);
        }

        yofs[dy]                   = sy;
        ybeta[(size_t)dy * 2 + 0] = CoefMake<CoefT>::make(1.f - fy);
        ybeta[(size_t)dy * 2 + 1] = CoefMake<CoefT>::make(fy);
    }
}

// -----------------------------------------------------------------------------
// One linear blend of the two float passes.
// Same ordering as OpenCV's v0*w0 + v1*w1; __fmul_rn/__fadd_rn are used
// explicitly to stop nvcc from contracting mul+add into an FMA -- OpenCV's SSE
// implementation multiplies separately and then adds, without fusion.
// -----------------------------------------------------------------------------
__device__ __forceinline__ float lerp2(float v0, float w0, float v1, float w1)
{
    return __fadd_rn(__fmul_rn(v0, w0), __fmul_rn(v1, w1));
}

// -----------------------------------------------------------------------------
// 8U -> 8U: fixed-point chain
//
// Thread mapping (same style as TensorRT-Alpha's preprocessing kernels):
//   x dimension (blockIdx.x/threadIdx.x) flattens all dstW*dstH output pixels of
//     one image
//   y dimension (blockIdx.y/threadIdx.y) iterates the batch index
// -----------------------------------------------------------------------------
template <int CN>
__global__ void resizeLinearU8Kernel(
    const unsigned char* __restrict__ src, int srcW, int srcH,
    unsigned char* __restrict__ dst, int dstW, int dstH,
    const int*   __restrict__ xofs,
    const short* __restrict__ xalpha,
    const int*   __restrict__ yofs,
    const short* __restrict__ ybeta,
    int batchSize)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // linear index of the output pixel
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch index

    if (dx >= dstW * dstH || dy >= batchSize)
    {
        return;
    }

    const int dstY = dx / dstW;
    const int dstX = dx - dstY * dstW;

    const unsigned char* srcImg = src + (size_t)dy * srcW * srcH * CN;
    unsigned char*       dstImg = dst + (size_t)dy * dstW * dstH * CN;

    // y clipping (sy may be negative)
    int syA = yofs[dstY];
    int syB = syA + 1;
    syA = syA < 0 ? 0 : (syA > srcH - 1 ? srcH - 1 : syA);
    syB = syB < 0 ? 0 : (syB > srcH - 1 ? srcH - 1 : syB);

    // x: clamp the second sample back to srcW-1 (a1 is necessarily 0 there, so the
    // result is unaffected, but it avoids an out-of-bounds read)
    const int sx0 = xofs[dstX];
    int       sx1 = sx0 + 1;
    if (sx1 > srcW - 1) sx1 = srcW - 1;

    const short a0 = xalpha[(size_t)dstX * 2 + 0];
    const short a1 = xalpha[(size_t)dstX * 2 + 1];
    const short b0 = ybeta [(size_t)dstY * 2 + 0];
    const short b1 = ybeta [(size_t)dstY * 2 + 1];

    const unsigned char* rowAp0 = srcImg + (size_t)syA * srcW * CN + (size_t)sx0 * CN;
    const unsigned char* rowAp1 = srcImg + (size_t)syA * srcW * CN + (size_t)sx1 * CN;
    const unsigned char* rowBp0 = srcImg + (size_t)syB * srcW * CN + (size_t)sx0 * CN;
    const unsigned char* rowBp1 = srcImg + (size_t)syB * srcW * CN + (size_t)sx1 * CN;
    unsigned char* out = dstImg + ((size_t)dstY * dstW + dstX) * CN;

#pragma unroll
    for (int c = 0; c < CN; ++c)
    {
        // horizontal convolution (int)
        const int rowA = (int)rowAp0[c] * a0 + (int)rowAp1[c] * a1;
        const int rowB = (int)rowBp0[c] * a0 + (int)rowBp1[c] * a1;

        // vertical convolution: step-by-step bit operations exactly as in OpenCV's
        // scalar code
        int val = (((b0 * (rowA >> 4)) >> 16)
                 + ((b1 * (rowB >> 4)) >> 16)
                 + 2) >> 2;

        if (val < 0)   val = 0;
        if (val > 255) val = 255;
        out[c] = (unsigned char)val;
    }
}

// -----------------------------------------------------------------------------
// 32F output: two float passes. TIn is either uchar (widened losslessly to float)
// or float.
// -----------------------------------------------------------------------------
template <int CN, typename TIn>
__global__ void resizeLinearF32Kernel(
    const TIn* __restrict__ src, int srcW, int srcH,
    float* __restrict__ dst, int dstW, int dstH,
    const int*   __restrict__ xofs,
    const float* __restrict__ xalpha,
    const int*   __restrict__ yofs,
    const float* __restrict__ ybeta,
    int batchSize)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;

    if (dx >= dstW * dstH || dy >= batchSize)
    {
        return;
    }

    const int dstY = dx / dstW;
    const int dstX = dx - dstY * dstW;

    const TIn* srcImg = src + (size_t)dy * srcW * srcH * CN;
    float*     dstImg = dst + (size_t)dy * dstW * dstH * CN;

    int syA = yofs[dstY];
    int syB = syA + 1;
    syA = syA < 0 ? 0 : (syA > srcH - 1 ? srcH - 1 : syA);
    syB = syB < 0 ? 0 : (syB > srcH - 1 ? srcH - 1 : syB);

    const int sx0 = xofs[dstX];
    int       sx1 = sx0 + 1;
    if (sx1 > srcW - 1) sx1 = srcW - 1;

    const float a0 = xalpha[(size_t)dstX * 2 + 0];
    const float a1 = xalpha[(size_t)dstX * 2 + 1];
    const float b0 = ybeta [(size_t)dstY * 2 + 0];
    const float b1 = ybeta [(size_t)dstY * 2 + 1];

    const TIn* rowAp0 = srcImg + (size_t)syA * srcW * CN + (size_t)sx0 * CN;
    const TIn* rowAp1 = srcImg + (size_t)syA * srcW * CN + (size_t)sx1 * CN;
    const TIn* rowBp0 = srcImg + (size_t)syB * srcW * CN + (size_t)sx0 * CN;
    const TIn* rowBp1 = srcImg + (size_t)syB * srcW * CN + (size_t)sx1 * CN;
    float* out = dstImg + ((size_t)dstY * dstW + dstX) * CN;

#pragma unroll
    for (int c = 0; c < CN; ++c)
    {
        const float rowA = lerp2((float)rowAp0[c], a0, (float)rowAp1[c], a1);
        const float rowB = lerp2((float)rowBp0[c], a0, (float)rowBp1[c], a1);
        out[c] = lerp2(rowA, b0, rowB, b1);
    }
}

// -----------------------------------------------------------------------------
// Issue one kernel (the only place on the host side that talks to the kernels)
// -----------------------------------------------------------------------------
template <typename TIn, typename TOut, int CN>
void launchOne(cudaStream_t stream, int batch,
               const TIn* src, TOut* dst,
               int srcW, int srcH, int dstW, int dstH,
               const int* d_xofs,
               const typename detail::CoefType<TOut>::type* d_xalpha,
               const int* d_yofs,
               const typename detail::CoefType<TOut>::type* d_ybeta)
{
    const int  pix   = dstW * dstH;
    const dim3 block(kBlock1D, 1);
    const dim3 grid((unsigned)((pix + kBlock1D - 1) / kBlock1D), (unsigned)batch);

    if constexpr (std::is_same<TOut, unsigned char>::value)
    {
        resizeLinearU8Kernel<CN><<<grid, block, 0, stream>>>(
            src, srcW, srcH, dst, dstW, dstH,
            d_xofs, d_xalpha, d_yofs, d_ybeta, batch);
    }
    else
    {
        resizeLinearF32Kernel<CN, TIn><<<grid, block, 0, stream>>>(
            src, srcW, srcH, dst, dstW, dstH,
            d_xofs, d_xalpha, d_yofs, d_ybeta, batch);
    }
}

}  // namespace

// =============================================================================
//  CvResizePlan
// =============================================================================
template <typename TIn, typename TOut>
CvResizePlan<TIn, TOut>::CvResizePlan(int srcW, int srcH, int dstW, int dstH)
    : srcW_(srcW), srcH_(srcH), dstW_(dstW), dstH_(dstH)
{
    if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    std::vector<int>   hxofs, hyofs;
    std::vector<CoefT> hxalpha, hybeta;
    buildCoeffs<CoefT>(srcW, srcH, dstW, dstH, hxofs, hxalpha, hyofs, hybeta);

    err_ = cudaMalloc(reinterpret_cast<void**>(&d_xofs_), (size_t)dstW * sizeof(int));
    if (err_ != cudaSuccess) return;
    err_ = cudaMalloc(reinterpret_cast<void**>(&d_xalpha_), (size_t)dstW * 2 * sizeof(CoefT));
    if (err_ != cudaSuccess) return;
    err_ = cudaMalloc(reinterpret_cast<void**>(&d_yofs_), (size_t)dstH * sizeof(int));
    if (err_ != cudaSuccess) return;
    err_ = cudaMalloc(reinterpret_cast<void**>(&d_ybeta_), (size_t)dstH * 2 * sizeof(CoefT));
    if (err_ != cudaSuccess) return;

    err_ = cudaMemcpy(d_xofs_,   hxofs.data(),   (size_t)dstW * sizeof(int),        cudaMemcpyHostToDevice);
    if (err_ != cudaSuccess) return;
    err_ = cudaMemcpy(d_xalpha_, hxalpha.data(), (size_t)dstW * 2 * sizeof(CoefT),  cudaMemcpyHostToDevice);
    if (err_ != cudaSuccess) return;
    err_ = cudaMemcpy(d_yofs_,   hyofs.data(),   (size_t)dstH * sizeof(int),        cudaMemcpyHostToDevice);
    if (err_ != cudaSuccess) return;
    err_ = cudaMemcpy(d_ybeta_,  hybeta.data(),  (size_t)dstH * 2 * sizeof(CoefT),  cudaMemcpyHostToDevice);
}

template <typename TIn, typename TOut>
CvResizePlan<TIn, TOut>::~CvResizePlan()
{
    release();
}

template <typename TIn, typename TOut>
CvResizePlan<TIn, TOut>::CvResizePlan(CvResizePlan&& other) noexcept
    : srcW_(other.srcW_), srcH_(other.srcH_), dstW_(other.dstW_), dstH_(other.dstH_),
      d_xofs_(other.d_xofs_), d_xalpha_(other.d_xalpha_),
      d_yofs_(other.d_yofs_), d_ybeta_(other.d_ybeta_),
      err_(other.err_)
{
    other.d_xofs_   = nullptr;
    other.d_xalpha_ = nullptr;
    other.d_yofs_   = nullptr;
    other.d_ybeta_  = nullptr;
}

template <typename TIn, typename TOut>
void CvResizePlan<TIn, TOut>::release() noexcept
{
    if (d_xofs_)   { cudaFree(d_xofs_);   d_xofs_   = nullptr; }
    if (d_xalpha_) { cudaFree(d_xalpha_); d_xalpha_ = nullptr; }
    if (d_yofs_)   { cudaFree(d_yofs_);   d_yofs_   = nullptr; }
    if (d_ybeta_)  { cudaFree(d_ybeta_);  d_ybeta_  = nullptr; }
}

template <typename TIn, typename TOut>
void CvResizePlan<TIn, TOut>::launch(cudaStream_t stream, int batch,
                                     const TIn* src, TOut* dst,
                                     int cn, Interp interp) const
{
    if (err_ != cudaSuccess)                        return;   // construction failed: see lastError()
    if (interp != Interp::Linear)                   { err_ = cudaErrorNotSupported; return; }
    if (src == nullptr || dst == nullptr)           { err_ = cudaErrorInvalidValue;   return; }
    if (batch <= 0)                                 return;
    if (batch > 65535)                              { err_ = cudaErrorInvalidValue;   return; }
    if (cn != 1 && cn != 3)                         { err_ = cudaErrorInvalidValue;   return; }
    if ((long long)dstW_ * dstH_ > 0x7fffffffLL)    { err_ = cudaErrorInvalidValue;   return; }

    switch (cn)
    {
        case 1: launchOne<TIn, TOut, 1>(stream, batch, src, dst,
                                        srcW_, srcH_, dstW_, dstH_,
                                        d_xofs_, d_xalpha_, d_yofs_, d_ybeta_);
                break;
        case 3: launchOne<TIn, TOut, 3>(stream, batch, src, dst,
                                        srcW_, srcH_, dstW_, dstH_,
                                        d_xofs_, d_xalpha_, d_yofs_, d_ybeta_);
                break;
        default: break;
    }

    // Fetch the error only, no sync (stay asynchronous)
    err_ = cudaGetLastError();
}

// =============================================================================
//  cvResize -- one-shot convenience version
// =============================================================================
template <typename TIn, typename TOut>
void cvResize(cudaStream_t stream, int batch,
              const TIn* src, int srcW, int srcH,
              TOut* dst, int dstW, int dstH,
              int cn, Interp interp)
{
    CvResizePlan<TIn, TOut> plan(srcW, srcH, dstW, dstH);
    plan.launch(stream, batch, src, dst, cn, interp);
}

// =============================================================================
//  Explicit instantiation (only these three combinations; others fail to link)
// =============================================================================
template class CvResizePlan<unsigned char, unsigned char>;
template class CvResizePlan<float, float>;
template class CvResizePlan<unsigned char, float>;

template void cvResize<unsigned char, unsigned char>(
    cudaStream_t, int, const unsigned char*, int, int, unsigned char*, int, int, int, Interp);
template void cvResize<float, float>(
    cudaStream_t, int, const float*, int, int, float*, int, int, int, Interp);
template void cvResize<unsigned char, float>(
    cudaStream_t, int, const unsigned char*, int, int, float*, int, int, int, Interp);

}  // namespace trt_alpha::kernels::ops
