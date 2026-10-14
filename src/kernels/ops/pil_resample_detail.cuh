// =============================================================================
//  trt_alpha :: kernels :: ops :: detail :: pil_resample_detail
// -----------------------------------------------------------------------------
//  Device-side helpers shared by all THREE Pillow-resample translation units
//      pil_resample.cu                       (the standalone operator)
//      preprocess_fused_yolov26_cls.cu       (fused, scalar)
//      preprocess_fused_vec_yolov26_cls.cu   (fused, vectorized)
//
//  Private to src/kernels/ops (not installed).  It is deliberately NOT a public
//  header: the three operators are independent translation units and must not
//  pull each other in, but the ported Pillow arithmetic has to be written exactly
//  once or the variants can drift -- the same reasoning behind
//  nms_utral_detail.cuh.
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/pil_resample.hpp"

#include <cuda_runtime.h>

namespace trt_alpha::kernels::ops::detail {

// -----------------------------------------------------------------------------
//! Pillow's clip8() -- Resample.c:177.
//!
//!     clip8_lookups = &_clip8_lookups[640];   // table: 640 zeros | 0..255 | 255s
//!     clip8(in)     = clip8_lookups[in >> PRECISION_BITS];
//!
//! For the valid input range the table collapses to a clamp of the arithmetic
//! shift.  `>>` on a signed int rounds toward minus infinity (which is what the
//! leading 640 zeros implement), so the clamp must be applied AFTER the shift.
//! The upper bound cannot be exceeded for 8-bit sources with non-negative
//! normalized coefficients, but it is kept so the function is total.
// -----------------------------------------------------------------------------
__device__ __forceinline__ unsigned char pilClip8(int v) noexcept
{
    const int i = v >> kPilPrecisionBits;
    return static_cast<unsigned char>(i < 0 ? 0 : (i > 255 ? 255 : i));
}

// -----------------------------------------------------------------------------
//! One output pixel of the HORIZONTAL pass -- Pillow _ImagingResampleHorizontal_8bpc,
//! bands == 3 branch (Resample.c:344).
//!
//! @param row   first byte of the source row (BGR, 3 bytes per pixel)
//! @param rx    output column of the RESIZED axis (bounds are indexed by it)
//! @param needH false -> the pass is skipped: the resized column rx IS source
//!              column rx, with no filtering and no clip8
//! @return      the clipped BGR triple of the resized image at (row, rx)
//!
//! The accumulator starts at 1 << (PRECISION_BITS - 1), not at 0: Pillow adds half
//! of the fixed-point LSB so that clip8's truncating shift rounds to nearest.
//! All arithmetic is int32; the sum cannot overflow for non-negative coefficients
//! because they are normalized (255 * 2^22 + 2^21 < 2^31).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uchar3 pilHorizPixel(const unsigned char* __restrict__ row,
                                               int rx, int needH,
                                               const PilBound* __restrict__ bH,
                                               const int* __restrict__ kH, int ksizeH) noexcept
{
    if (!needH)
    {
        const unsigned char* p = row + static_cast<std::size_t>(rx) * 3;
        return make_uchar3(p[0], p[1], p[2]);
    }

    const PilBound b = bH[rx];
    const int* k = kH + static_cast<std::size_t>(rx) * ksizeH;
    const unsigned char* p = row + static_cast<std::size_t>(b.first) * 3;

    int s0 = kPilRoundHalf, s1 = kPilRoundHalf, s2 = kPilRoundHalf;
    for (int t = 0; t < b.count; ++t, p += 3)
    {
        const int kt = k[t];
        s0 += p[0] * kt;
        s1 += p[1] * kt;
        s2 += p[2] * kt;
    }
    return make_uchar3(pilClip8(s0), pilClip8(s1), pilClip8(s2));
}

}  // namespace trt_alpha::kernels::ops::detail
