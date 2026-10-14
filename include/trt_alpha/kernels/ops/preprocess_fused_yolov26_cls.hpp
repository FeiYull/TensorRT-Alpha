// =============================================================================
//  trt_alpha :: kernels :: ops :: preprocess_fused_yolov26_cls
// -----------------------------------------------------------------------------
//  Three-in-one: fuses the cls preprocessing chain into a SINGLE kernel.
//
//      Resize (shortest edge -> outW, BILINEAR)   Pillow libImaging
//      CenterCrop (outW x outH)                   torchvision
//      BGR -> RGB                                 numpy img[..., ::-1]
//      HWC -> CHW                                 numpy transpose(2,0,1)
//      /255 (float)                               torch .float().div_(255.0)
//
//  Input : uchar*  BGR HWC  single image [srcH][srcW][3]   batch back to back
//  Output: float*  RGB CHW  single image [3][outH][outW]   batch back to back
//                                                        (fed to the network)
//
//  This is the cls counterpart of preprocess_fused_yolov26_det, and it is a
//  SEPARATE operator rather than a mode of it because the geometry has nothing in
//  common: detection/segmentation/pose/obb/depth/semantic letterbox the image
//  (uniform scale + 114 grey border, keeping aspect), while classification
//  resizes the shortest edge to the network size and then CROPS the longer edge.
//  There is no padding anywhere in this chain, so there is no 114 and no pad value.
//
//  ------------------------------------------------------------- why one kernel
//  Pillow's resample is a two-pass separable filter whose horizontal result is
//  clipped to uint8 before the vertical pass reads it.  That makes a fully fused
//  kernel legal: for each output pixel the horizontal value of a tap row can be
//  recomputed on the fly and it is bit-identical to the value a materialized
//  intermediate would have held, because every step is integer (see
//  yolov26_cls_common.hpp).  Fusing also removes the intermediate buffer, which
//  for an extreme aspect ratio would otherwise be much larger than the output.
//
//  ------------------------------------------------- fixed output size (static ONNX)
//  The engine input is a static ONNX shape, so outW/outH are fixed.  The chain
//  still has TWO different axis behaviours, which is why needH / needV exist:
//      needV = newH != srcH, needH = newW != srcW
//  When a flag is false that axis is copied verbatim with no filtering at all --
//  this is the "640x640 in, 640x640 out" case where torchvision's Resize returns
//  the image unchanged and the crop is the identity.
//
//  WARNING: this differs from the auto=True/rect letterbox used in the Python
//  reference pipeline, exactly as preprocess_fused_yolov26_det does -- a static
//  engine cannot take a stride-aligned rectangle.  The two therefore agree only
//  on images whose geometry coincides (square sources); see the project notes.
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/pil_resample.hpp"
#include "trt_alpha/kernels/ops/yolov26_cls_common.hpp"

#include <cuda_runtime.h>   // cudaStream_t / cudaError_t / cudaSuccess

#include <cstddef>
#include <memory>

namespace trt_alpha::kernels::ops {

// =============================================================================
//  CvPreprocessClsPlan -- geometry and coefficient tables fixed in the
//  constructor, only launch() in the loop.
// -----------------------------------------------------------------------------
//  The constructor computes the geometry (computePilClsGeometry), builds the two
//  fixed-point coefficient tables (buildPilResampleAxis) and uploads them to the
//  device.  Afterwards launch() neither allocates, nor copies, nor syncs.
//
//  Images in one batch must share geometry (same srcW/srcH); rebuild the plan
//  when the geometry changes.
//
//  Usage:
//      auto plan = CvPreprocessClsPlan::byDivisor(1280, 1280, 640, 640);
//      for (...) plan.launch(stream, batch, d_src, d_dst);   // d_dst is float*
// =============================================================================
class CvPreprocessClsPlan
{
public:
    CvPreprocessClsPlan(int srcW, int srcH, int outW, int outH, float divisor = 255.0f);

    //! dst = src / divisor -- bit-exact with torch's uint8.float().div_(255.0)
    static CvPreprocessClsPlan byDivisor(int srcW, int srcH, int outW, int outH,
                                         float divisor = 255.0f);

    ~CvPreprocessClsPlan();

    CvPreprocessClsPlan(const CvPreprocessClsPlan&)            = delete;
    CvPreprocessClsPlan& operator=(const CvPreprocessClsPlan&) = delete;
    CvPreprocessClsPlan(CvPreprocessClsPlan&& other) noexcept;

    //! Does the geometry match this plan? (A mismatch must be rebuilt.)
    bool matches(int srcW, int srcH, int outW, int outH) const noexcept
    {
        return cfg_.srcW == srcW && cfg_.srcH == srcH
            && cfg_.outW == outW && cfg_.outH == outH;
    }

    //! Geometry (source / resized / crop sizes and the two need* flags).
    const PreprocessPilClsConfig& info() const noexcept { return cfg_; }

    int    outW() const noexcept { return cfg_.outW; }
    int    outH() const noexcept { return cfg_.outH; }
    //! Output elements per image = 3*outW*outH
    std::size_t outElems() const noexcept
    {
        return static_cast<std::size_t>(cfg_.outW) * cfg_.outH * 3;
    }

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel.
    //!   batch : number of images; src must hold batch images of srcW*srcH*3
    //!           uchar, dst must hold batch * 3*outW*outH floats.
    //! No allocation, no copy, no sync.
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const unsigned char* src, float* dst) const;

private:
    PreprocessPilClsConfig cfg_{};

    //! Only its fixed-point coefficient tables are borrowed -- the same tables the
    //! standalone CvPilResamplePlan is verified against, so the individual and
    //! fused paths cannot drift. CvPilResamplePlan is not assignable, hence the
    //! unique_ptr (the same shape as CvPreprocessPlan's CvResizePlan member).
    std::unique_ptr<CvPilResamplePlan> resample_;

    float divisor_ = 255.0f;

    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  preprocessFusedClsDevice -- one-shot convenience version
// =============================================================================
//  Creates and destroys internally (rebuilds the geometry and re-uploads the
//  coefficient tables every call).  Use it for a single call or in tests; the hot
//  loop should use CvPreprocessClsPlan.
//
//  The default is equivalent to dst = src / 255.
// =============================================================================
void preprocessFusedClsDevice(cudaStream_t stream, int batch,
                              const unsigned char* src, float* dst,
                              int srcW, int srcH, int outW, int outH,
                              float divisor = 255.0f);

}  // namespace trt_alpha::kernels::ops
