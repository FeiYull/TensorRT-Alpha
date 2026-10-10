// =============================================================================
//  trt_alpha :: kernels :: ops :: numeric
// -----------------------------------------------------------------------------
//  Numeric helpers (rounding / saturating casts, OpenCV-equivalent) plus the
//  shared conventions for the preprocessing operators (resize / copyMakeBorder
//  / cvtColor / hwc2chw / normalize) and their fused variants. Every new
//  operator follows the memory layout and naming rules below.
//
//  Memory layout (src / dst are device pointers, identical to resize):
//      single image  [H][W][cn]   (HWC row-major, channels interleaved, i.e. a
//                                  contiguous cv::Mat block)
//      batch         images are packed back to back:
//                    start of image n = base + n * H * W * cn
//
//  Channel count: cn supports only 1 / 3 (same rule as resize).
//
//  Naming (same lineage as TensorRT-Alpha's hwcToChwKernel / hwcToChw):
//      host wrapper = action + "Device" suffix
//                     copyMakeBorderDevice / cvtColorDevice / hwc2chwDevice /
//                     normalizeDevice
//      Plan class   = action + "Plan"
//                     CvCopyMakeBorderPlan / CvCvtColorPlan /
//                     CvHwcToChwPlan / CvNormalizePlan
//      each operator = one .hpp + one .cu; CMake collects src/*.cu with
//                     GLOB_RECURSE, so adding a file needs no CMake edit.
//
//  Why every operator offers both a "Plan" and a one-shot interface:
//      to keep one single usage pattern with CvResizePlan -- construct outside
//      the loop (precompute / pre-upload), launch only inside the loop (no
//      allocation, no copy, no sync). Geometry is fixed for a video stream, so
//      the whole lifecycle constructs it once.
//
//  Error handling: never throws, never syncs. Records err_ only; query it with
//  ok() / lastError().
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace trt_alpha::kernels::ops {

// NOTE: the 1D launch block size is trt_alpha::kernels::kBlock1D (= 256),
//       declared in trt_alpha/kernels/common.cuh and shared with the other
//       kernels. The standalone project used to carry its own duplicate
//       "kThreads1D" here; it is gone -- use kBlock1D instead.

namespace detail
{

//! Round to nearest, ties to even -- equivalent to OpenCV's cvRound
//! (SSE cvtsd2si uses the default rounding mode).
inline int roundHalfToEven(double v)
{
    return (int)std::nearbyint(v);
}

//! Equivalent to cv::saturate_cast<unsigned char>(double): cvRound + clamp to [0,255].
inline unsigned char saturateCastU8(double v)
{
    int iv = roundHalfToEven(v);
    if (iv < 0)   iv = 0;
    if (iv > 255) iv = 255;
    return (unsigned char)iv;
}

//! Equivalent to cv::saturate_cast<float>(double): plain narrowing cast
//! (no rounding, no clamping).
inline float saturateCastF32(double v)
{
    return (float)v;
}

//! Turn a double constant into the target pixel type -- used for the
//! BORDER_CONSTANT Scalar fill value.
template <typename T> struct CastFill;
template <> struct CastFill<unsigned char>
{
    static unsigned char make(double v) { return saturateCastU8(v); }
};
template <> struct CastFill<float>
{
    static float make(double v) { return saturateCastF32(v); }
};

//! Validation: cn may only be 1 / 3 (same rule as resize).
inline bool validChannels(int cn) { return cn == 1 || cn == 3; }

}  // namespace detail
}  // namespace trt_alpha::kernels::ops
