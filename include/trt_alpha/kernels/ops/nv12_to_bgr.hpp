// =============================================================================
//  trt_alpha :: kernels :: ops :: nv12_to_bgr
// -----------------------------------------------------------------------------
//  NV12 -> packed BGR8, entirely on the device.
//
//  Why this exists: NVDEC hands out NV12 in device memory (two planes, two
//  pitches). Downloading it and letting libswscale convert would put a 1080p
//  colour conversion (2-4 ms) back on the CPU, which is exactly what the
//  hardware decoder is supposed to remove. The whole point of the FFmpeg
//  backend is that the CPU never touches a pixel.
//
//  The matrix / range are passed in because NVDEC output follows the stream's
//  colour metadata: BT.601 vs BT.709 and limited vs full range differ by up to
//  ~2 % on saturated colours, which is visible and would break the "same frames
//  as OpenCV" comparison.
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace trt_alpha::kernels::ops {

//! Colour matrix (mirrors AVColorSpace, reduced to the ones that matter here).
enum class YuvMatrix : std::uint8_t
{
    BT601,    //!< kr = 0.2990, kb = 0.1140 (SD, "SMPTE 170M" / "BT.470BG")
    BT709,    //!< kr = 0.2126, kb = 0.0722 (HD)
    BT2020,   //!< kr = 0.2627, kb = 0.0593 (UHD / HDR)
};

//! Sample range (mirrors AVColorRange).
enum class YuvRange : std::uint8_t
{
    Limited,  //!< Y in [16,235], C in [16,240] -- broadcast / camera default
    Full,     //!< Y in [0,255], C in [0,255]   -- JPEG / screen capture
};

//! Convert one NV12 image to packed BGR8.
//!
//!   y / uv            device pointers (AVFrame::data[0] / data[1])
//!   yPitch / uvPitch  bytes per row of each plane (AVFrame::linesize[0] / [1])
//!   bgr / bgrPitch    destination; bgrPitch must be >= width * 3
//!   width / height    luma dimensions; must both be even (4:2:0)
//!
//! Launches on `stream` and returns cudaGetLastError() -- no synchronisation.
[[nodiscard]] cudaError_t nv12ToBgr(cudaStream_t stream,
                                    const std::uint8_t* y,
                                    const std::uint8_t* uv,
                                    int yPitch,
                                    int uvPitch,
                                    std::uint8_t* bgr,
                                    int bgrPitch,
                                    int width,
                                    int height,
                                    YuvMatrix matrix,
                                    YuvRange range);

}  // namespace trt_alpha::kernels::ops
