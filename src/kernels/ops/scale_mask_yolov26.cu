// =============================================================================
//  trt_alpha :: kernels :: ops :: scale_mask_yolov26  (implementation)
// -----------------------------------------------------------------------------
//  Host side : geometry (reusing computeLetterBox -- the single source of truth
//              shared with the forward letterbox).
//  Device side: one kernel; one thread = one output pixel of one plane.
//
//  Host geometry comes from ultralytics
//      ultralytics/utils/ops.py :: scale_masks
//          im1_h, im1_w = masks.shape[2:]        # network-space mask
//          im0_h, im0_w = shape[:2]              # target image
//          gain_h = gain_w = min(im1_h/im0_h, im1_w/im0_w)
//          pad_w = (im1_w - round(im0_w * gain_w)) / 2
//          pad_h = (im1_h - round(im0_h * gain_h)) / 2
//          top, left = round(pad_h - 0.1), round(pad_w - 0.1)
//          bottom = top + round(im0_h * gain_h)
//          right  = left + round(im0_w * gain_w)
//          F.interpolate(masks[..., top:bottom, left:right], shape, mode=mode)
//  which is the very same window computeLetterBox() derives -- one truth.
//
//  Device maths is a port of the ATen CPU kernel that actually runs for a
//  float mask (C == 1), namely
//      aten/src/ATen/native/cpu/UpSampleKernel.cpp
//        upsample_bilinear2d_kernel_impl_float        :1865
//          -> upsample_non_separable_Nd_kernel_impl   :1511
//               -> InterpolateNonSeparable<2,..,2>::eval  :120
//                    t0  = i00 * w0  + i01 * w1
//                    t1  = i10 * w0  + i11 * w1
//                    out = t0  * h0  + t1  * h1      <-- exactly our grouping
//  with, from aten/src/ATen/native/UpSample.h:
//      compute_scales_value<float>(nullopt, in, out) :261   -> (float)in / out
//      area_pixel_compute_source_index<float>(..)    :291
//                              scale * (dst + 0.5) - 0.5, clamped to >= 0
//      guard_index_and_lambda(..)                    :441
//                              idx = min(floorf(idx), in-1),
//                              lam = clamp(idx_real - idx, 0, 1)
//      nearest_neighbor_compute_source_index         :318
//                              min(floorf(dst * scale), in-1)
//
//  Verified bit-exact against torch 2.14.0 (sample_scale_mask_compare,
//  11/12 geometries max|d| = 0 on bilinear and 12/12 on nearest). The one
//  exception is a degenerate 3x7 output: ATen's dispatcher
//      _use_channels_last_kernel_2d()                 :1765
//      -> (outH + outW) <= 128  ==>  upsample_linear_channels_last()
//  then uses the *other* kernel, whose weights are pre-multiplied
//  (w00 = h0*w0, ...) and summed in a different association. That path is
//  unreachable for any real image (outH + outW > 128), so we deliberately
//  keep the single non-separable grouping rather than carry a second kernel.
// =============================================================================
#include "trt_alpha/kernels/ops/scale_mask_yolov26.hpp"

#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>

namespace trt_alpha::kernels::ops {
namespace {

// =============================================================================
//  Device kernel
// -----------------------------------------------------------------------------
//  Thread mapping: x flattens the srcW*srcH output pixels, y is the plane index
//  (batch * channels + channel). Output writes are fully coalesced.
// =============================================================================
template <typename T, bool kNearest>
__global__ void scaleMaskKernel(
    const T* __restrict__ src,
    int netW, int netH,
    int cropLeft, int cropTop, int newW, int newH,
    T* __restrict__ dst,
    int outW, int outH)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outW * outH)
    {
        return;
    }

    const int oy = idx / outW;
    const int ox = idx - oy * outW;

    const float rh = static_cast<float>(newH) / static_cast<float>(outH);
    const float rw = static_cast<float>(newW) / static_cast<float>(outW);

    const T* in  = src + static_cast<std::size_t>(blockIdx.y) *
                         static_cast<std::size_t>(netW) * netH;
    T*       out = dst + static_cast<std::size_t>(blockIdx.y) *
                         static_cast<std::size_t>(outW) * outH;

    if (kNearest)
    {
        // torch's non-exact nearest: src = min(floor(dst * in/out), in-1)
        int y = static_cast<int>(floorf(static_cast<float>(oy) * rh));
        int x = static_cast<int>(floorf(static_cast<float>(ox) * rw));
        if (y > newH - 1) { y = newH - 1; }
        if (x > newW - 1) { x = newW - 1; }
        out[idx] = in[static_cast<std::size_t>(cropTop + y) * netW + (cropLeft + x)];
        return;
    }

    // Bilinear, align_corners=false (torch's default). The clamping at 0 mirrors
    // area_pixel_compute_source_index(..., cubic=false).
    float sy = __fadd_rn(__fmul_rn(rh, __fadd_rn(static_cast<float>(oy), 0.5f)), -0.5f);
    float sx = __fadd_rn(__fmul_rn(rw, __fadd_rn(static_cast<float>(ox), 0.5f)), -0.5f);
    if (sy < 0.f) { sy = 0.f; }
    if (sx < 0.f) { sx = 0.f; }

    const int y0 = static_cast<int>(sy);
    const int x0 = static_cast<int>(sx);
    const int y1 = (y0 + 1 < newH) ? y0 + 1 : y0;
    const int x1 = (x0 + 1 < newW) ? x0 + 1 : x0;

    const float h1l = sy - static_cast<float>(y0);
    const float h0l = 1.f - h1l;
    const float w1l = sx - static_cast<float>(x0);
    const float w0l = 1.f - w1l;

    const T* r0 = in + static_cast<std::size_t>(cropTop + y0) * netW + cropLeft;
    const T* r1 = in + static_cast<std::size_t>(cropTop + y1) * netW + cropLeft;

    // __fmul_rn / __fadd_rn are NEVER contracted into an FMA by nvcc, which
    // matters here: the reference for this project is torch on CPU (the yolo26
    // conda env has no CUDA build), and torch's CPU kernel does plain mul + add.
    // Letting nvcc fuse "h0l*t0 + h1l*t1" into an FMA is *more* accurate but
    // lands 1 ULP away from the reference (measured 4.8e-7 at value ~4.8, i.e.
    // exactly 1 ULP). Keep the operations separate.
    const float t0 = __fadd_rn(__fmul_rn(w0l, static_cast<float>(r0[x0])),
                               __fmul_rn(w1l, static_cast<float>(r0[x1])));
    const float t1 = __fadd_rn(__fmul_rn(w0l, static_cast<float>(r1[x0])),
                               __fmul_rn(w1l, static_cast<float>(r1[x1])));

    out[idx] = static_cast<T>(__fadd_rn(__fmul_rn(h0l, t0), __fmul_rn(h1l, t1)));
}

}  // namespace

// =============================================================================
//  CvScaleMaskPlan
// =============================================================================
template <typename T>
CvScaleMaskPlan<T>::CvScaleMaskPlan(int srcW, int srcH, int netW, int netH,
                                    MaskInterp mode, bool padding)
    : mode_(mode)
{
    if (srcW <= 0 || srcH <= 0 || netW <= 0 || netH <= 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }
    if (mode == MaskInterp::Bilinear && sizeof(T) != sizeof(float))
    {
        // Bilinear on integer data would invent values (e.g. class ids).
        TRT_LOG_ERROR("CvScaleMaskPlan: bilinear is only supported for float masks");
        err_ = cudaErrorInvalidValue;
        return;
    }

    info_.srcW = srcW;
    info_.srcH = srcH;
    info_.netW = netW;
    info_.netH = netH;

    if (padding)
    {
        // Same geometry function the forward letterbox uses -> single truth.
        const LetterBoxInfo lb = computeLetterBox(srcW, srcH, netW, netH);
        if (lb.newW < 1 || lb.newH < 1 ||
            lb.padLeft < 0 || lb.padTop < 0 ||
            lb.padLeft + lb.newW > netW || lb.padTop + lb.newH > netH)
        {
            err_ = cudaErrorInvalidValue;
            return;
        }
        info_.newW     = lb.newW;
        info_.newH     = lb.newH;
        info_.cropLeft = lb.padLeft;
        info_.cropTop  = lb.padTop;
    }
    else
    {
        // scale_masks(padding=False): no padding was applied to the mask.
        info_.newW     = netW;
        info_.newH     = netH;
        info_.cropLeft = 0;
        info_.cropTop  = 0;
    }

    err_ = cudaSuccess;
}

template <typename T>
void CvScaleMaskPlan<T>::launch(cudaStream_t stream, int batch,
                                const T* src, T* dst, int channels) const
{
    if (err_ != cudaSuccess)
    {
        return;
    }
    if (src == nullptr || dst == nullptr || batch <= 0 || channels <= 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }
    const int planes = batch * channels;
    if (planes > 65535)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    const int        n  = info_.srcW * info_.srcH;
    const dim3       grid((n + kBlock1D - 1) / kBlock1D, planes);
    const MaskInterp m = mode_;

    if (m == MaskInterp::Nearest)
    {
        scaleMaskKernel<T, true><<<grid, kBlock1D, 0, stream>>>(
            src, info_.netW, info_.netH, info_.cropLeft, info_.cropTop,
            info_.newW, info_.newH, dst, info_.srcW, info_.srcH);
    }
    else
    {
        scaleMaskKernel<T, false><<<grid, kBlock1D, 0, stream>>>(
            src, info_.netW, info_.netH, info_.cropLeft, info_.cropTop,
            info_.newW, info_.newH, dst, info_.srcW, info_.srcH);
    }

    err_ = cudaGetLastError();
    if (err_ != cudaSuccess)
    {
        TRT_LOG_ERROR("CvScaleMaskPlan: kernel launch failed: "
                      << cudaGetErrorString(err_));
    }
}

template <typename T>
void scaleMaskDevice(cudaStream_t stream, int batch,
                     const T* src, T* dst, int channels,
                     int srcW, int srcH, int netW, int netH,
                     MaskInterp mode)
{
    CvScaleMaskPlan<T> plan(srcW, srcH, netW, netH, mode);
    plan.launch(stream, batch, src, dst, channels);
    if (!plan.ok())
    {
        TRT_LOG_ERROR("scaleMaskDevice: " << cudaGetErrorString(plan.lastError()));
    }
}

// ---- explicit instantiations ------------------------------------------------
template class CvScaleMaskPlan<float>;
template class CvScaleMaskPlan<unsigned char>;
template void scaleMaskDevice<float>(cudaStream_t, int, const float*, float*, int,
                                     int, int, int, int, MaskInterp);
template void scaleMaskDevice<unsigned char>(cudaStream_t, int, const unsigned char*,
                                             unsigned char*, int, int, int, int, int,
                                             MaskInterp);

}  // namespace trt_alpha::kernels::ops
