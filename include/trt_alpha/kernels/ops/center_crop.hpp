// =============================================================================
//  trt_alpha :: kernels :: ops :: center_crop
// -----------------------------------------------------------------------------
//  CUDA-C reproduction of torchvision's `transforms.CenterCrop(size)` as it is
//  applied to a PIL image (torchvision/transforms/functional.py:592):
//
//      top  = int(round((image_height - crop_height) / 2.0))
//      left = int(round((image_width  - crop_width ) / 2.0))
//      img[top : top + crop_height, left : left + crop_width]
//
//  Two details that a from-memory implementation always gets wrong:
//    * the offsets go through Python's round(), which is HALF-TO-EVEN, not
//      half-away-from-zero. On a 320x240 source upscaled to 853x640 the crop is
//      106 from the left (round(106.5) == 106), not 107.
//    * the crop is clamped to the image by construction; CenterCrop would PAD
//      with zeros when the image is smaller than the crop, and that path is NOT
//      implemented here -- the plan records cudaErrorInvalidValue instead of
//      silently producing a wrong result.
//
//  Pure data movement, zero arithmetic, so 8U must be bit-exact.
//
//  Memory layout (same as the other preprocessing operators):
//      single image  [H][W][3]  HWC interleaved
//      batch         images are packed back to back
//      src / dst are device pointers
//
//  Only cn == 3 is supported (the cls chain has no other use for it); any other
//  cn records cudaErrorInvalidValue.
//
//  Error handling follows numeric.hpp: never throws, never syncs; records err_
//  only, query it with ok() / lastError().
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace trt_alpha::kernels::ops {

// =============================================================================
//  CenterCropInfo -- where the window starts and how big it is
// =============================================================================
struct CenterCropInfo
{
    int srcW  = 0, srcH  = 0;   //!< input size
    int cropW = 0, cropH = 0;   //!< window size
    int left  = 0, top   = 0;   //!< window origin, from Python's round() (half-to-even)

    bool valid() const noexcept
    {
        return srcW > 0 && srcH > 0 && cropW > 0 && cropH > 0
            && cropW <= srcW && cropH <= srcH;
    }
};

// =============================================================================
//! The geometry only -- mirrors the two `int(round(...))` lines above.
//!
//! Exported because the fused cls preprocessing needs exactly the same offsets,
//! and a second copy would drift. Implemented in center_crop.cu.
// =============================================================================
CenterCropInfo computeCenterCrop(int srcW, int srcH, int cropW, int cropH);

// =============================================================================
//  CvCenterCropPlan -- geometry fixed in the constructor, only launch in the loop
// -----------------------------------------------------------------------------
//  cvtColor/hwc2chw have nothing to precompute either, but they keep the same
//  pattern so the whole operator family looks alike:
//      CvCenterCropPlan plan(newW, newH, outW, outH);
//      for (...) plan.launch(stream, batch, d_src, d_dst);
// =============================================================================
class CvCenterCropPlan
{
public:
    //! srcW x srcH is the input, cropW x cropH the extracted window. Failure is
    //! reported via ok() / lastError(), not by throwing.
    CvCenterCropPlan(int srcW, int srcH, int cropW, int cropH);
    ~CvCenterCropPlan();

    CvCenterCropPlan(const CvCenterCropPlan&)            = delete;
    CvCenterCropPlan& operator=(const CvCenterCropPlan&) = delete;
    CvCenterCropPlan(CvCenterCropPlan&& other) noexcept;

    bool matches(int srcW, int srcH, int cropW, int cropH) const noexcept
    {
        return info_.srcW == srcW && info_.srcH == srcH
            && info_.cropW == cropW && info_.cropH == cropH;
    }

    const CenterCropInfo& info() const noexcept { return info_; }

    int cropW() const noexcept { return info_.cropW; }
    int cropH() const noexcept { return info_.cropH; }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    //! Hot path: issues exactly one kernel.
    //!   batch : number of images; src/dst must hold batch images each.
    //!   cn    : channel count, only 3
    void launch(cudaStream_t stream, int batch,
                const unsigned char* src, unsigned char* dst, int cn = 3) const;

private:
    CenterCropInfo info_;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  centerCropDevice -- one-shot convenience version
// =============================================================================
void centerCropDevice(cudaStream_t stream, int batch,
                      const unsigned char* src, unsigned char* dst,
                      int srcW, int srcH, int cropW, int cropH, int cn = 3);

}  // namespace trt_alpha::kernels::ops
