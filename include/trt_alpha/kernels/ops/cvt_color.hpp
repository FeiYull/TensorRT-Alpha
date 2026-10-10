// =============================================================================
//  trt_alpha :: kernels :: ops :: cvt_color
// -----------------------------------------------------------------------------
//  CUDA-C reproduction of OpenCV 4.8.1's cv::cvtColor(..., COLOR_BGR2RGB).
//
//  OpenCV reference implementation (modules/imgproc/src/color.cpp):
//      COLOR_BGR2RGB (enum value 4)
//        -> swapBlue(_src_cn = 3)
//        -> mixChannels(fromTo = {2, 1, 0})
//      i.e. dst[c] = src[fromTo[c]]: pure channel reordering, no arithmetic.
//
//  Therefore both 8U and 32F must be bit-exact (the test requires max = 0).
//
//  Only cn == 3 is supported -- the definition of BGR2RGB itself requires 3
//  channels; any other cn returns cudaErrorInvalidValue.
//
//  Memory layout (same as resize): [batch][H][W][cn], src/dst are device
//  pointers.
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/numeric.hpp"

namespace trt_alpha::kernels::ops {

//! Colour-space conversion mode. Only BGR2RGB is implemented in this round.
enum class ColorCode
{
    //! Implemented. Matches cv::COLOR_BGR2RGB.
    //! Note that cv::COLOR_RGB2BGR has the same enum value 4 (with 3 channels
    //! the two are fully equivalent), so this implementation covers both codes.
    BGR2RGB  = 4,

    BGR2GRAY = 6,    //!< TODO (match cv::COLOR_BGR2GRAY: Y = 0.299R + 0.587G + 0.114B)
    RGB2GRAY = 7,    //!< TODO
    GRAY2BGR = 8,    //!< TODO
    BGR2BGRA = 0,    //!< TODO
    BGRA2BGR = 1,    //!< TODO
    RGBA2BGR = 3     //!< TODO
};

// =============================================================================
//  CvCvtColorPlan
// -----------------------------------------------------------------------------
//  cvtColor has no table to precompute (BGR2RGB is a pure channel reorder), so
//  the Plan only fixes (width, height, code). It is kept so the usage pattern
//  matches CvResizePlan:
//      CvCvtColorPlan<unsigned char, unsigned char> plan(w, h, ColorCode::BGR2RGB);
//      for (...) plan.launch(stream, batch, d_src, d_dst, 3);
// =============================================================================
template <typename TIn, typename TOut>
class CvCvtColorPlan
{
public:
    CvCvtColorPlan(int width, int height, ColorCode code = ColorCode::BGR2RGB);
    ~CvCvtColorPlan();

    CvCvtColorPlan(const CvCvtColorPlan&)            = delete;
    CvCvtColorPlan& operator=(const CvCvtColorPlan&) = delete;
    CvCvtColorPlan(CvCvtColorPlan&& other) noexcept;

    bool matches(int width, int height, ColorCode code) const noexcept
    {
        return width_ == width && height_ == height && code_ == code;
    }

    int       width()  const noexcept { return width_; }
    int       height() const noexcept { return height_; }
    ColorCode code()   const noexcept { return code_; }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    //! Hot path: issues exactly one kernel. cn must be 3.
    void launch(cudaStream_t stream, int batch,
                const TIn* src, TOut* dst, int cn) const;

private:
    int       width_ = 0, height_ = 0;
    ColorCode code_  = ColorCode::BGR2RGB;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  cvtColorDevice -- one-shot convenience version
// =============================================================================
template <typename TIn, typename TOut>
void cvtColorDevice(cudaStream_t stream, int batch,
                    const TIn* src, TOut* dst,
                    int width, int height, int cn,
                    ColorCode code = ColorCode::BGR2RGB);

}  // namespace trt_alpha::kernels::ops
