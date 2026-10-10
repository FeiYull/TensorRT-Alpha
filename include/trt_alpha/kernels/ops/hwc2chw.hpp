// =============================================================================
//  trt_alpha :: kernels :: ops :: hwc2chw
// -----------------------------------------------------------------------------
//  HWC -> CHW layout conversion, equivalent to numpy's:
//      img_chw = np.ascontiguousarray(img.transpose(2, 0, 1))
//
//  Multi-batch version:
//      src  [batch][H][W][C]   ->   dst  [batch][C][H][W]
//
//  This is not an OpenCV function (OpenCV does it through combinations such as
//  cv::dnn::blobFromImage), but it is a mandatory step of inference
//  preprocessing. The naming follows TensorRT-Alpha's hwcToChw.
//
//  Pure data movement, zero arithmetic -- both 8U and 32F must be bit-exact
//  (the test requires max = 0).
//
//  C supports only 1 / 3:
//      C == 1  HWC and CHW are isomorphic, so it is a plain copy (1D grid)
//      C == 3  read 3 consecutive HWC channels and write them into 3 planes
//              (each thread reads 3 consecutive elements and writes 3 strided
//               planes; both reads and writes are coalesced, so no shared-memory
//               tiling is needed)
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/numeric.hpp"

namespace trt_alpha::kernels::ops {

// =============================================================================
//  CvHwcToChwPlan
// -----------------------------------------------------------------------------
//  (width, height, channels) all go into the constructor -- channels determines
//  the dst stride and is part of the geometry, unlike resize's cn which can stay
//  in launch().
//
//      CvHwcToChwPlan<float, float> plan(w, h, 3);
//      for (...) plan.launch(stream, batch, d_src, d_dst);
// =============================================================================
template <typename TIn, typename TOut>
class CvHwcToChwPlan
{
public:
    CvHwcToChwPlan(int width, int height, int channels);
    ~CvHwcToChwPlan();

    CvHwcToChwPlan(const CvHwcToChwPlan&)            = delete;
    CvHwcToChwPlan& operator=(const CvHwcToChwPlan&) = delete;
    CvHwcToChwPlan(CvHwcToChwPlan&& other) noexcept;

    bool matches(int width, int height, int channels) const noexcept
    {
        return width_ == width && height_ == height && channels_ == channels;
    }

    int width()    const noexcept { return width_; }
    int height()   const noexcept { return height_; }
    int channels() const noexcept { return channels_; }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    //! Hot path: issues exactly one kernel.
    //!   batch : number of images; src/dst must hold batch images each.
    void launch(cudaStream_t stream, int batch, const TIn* src, TOut* dst) const;

private:
    int width_ = 0, height_ = 0, channels_ = 0;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  hwc2chwDevice -- one-shot convenience version
// =============================================================================
template <typename TIn, typename TOut>
void hwc2chwDevice(cudaStream_t stream, int batch,
                   const TIn* src, TOut* dst,
                   int width, int height, int channels);

}  // namespace trt_alpha::kernels::ops
