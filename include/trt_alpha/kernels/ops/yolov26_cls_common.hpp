// =============================================================================
//  trt_alpha :: kernels :: ops :: yolov26_cls_common
// -----------------------------------------------------------------------------
//  The config struct shared by the two FUSED YOLOV26-cls preprocessing
//  operators
//      preprocess_fused_yolov26_cls.hpp        (scalar)
//      preprocess_fused_vec_yolov26_cls.hpp    (vectorized)
//
//  It lives HERE rather than in one of the two operator headers so that neither
//  has to include the other just to get the parameters, and there is exactly ONE
//  definition -- the same reasoning that produced yolov26_det_common.hpp.
//
//  The individual operators of the same chain are separate, self-contained
//  files:
//      pil_resample.hpp/.cu    Pillow BILINEAR resize            (the geometry
//                              constants and coefficient tables live there)
//      center_crop.hpp/.cu     torchvision CenterCrop
//      cvt_color.hpp/.cu       BGR -> RGB          (reused, OpenCV port)
//      hwc2chw.hpp/.cu         HWC -> CHW          (reused)
//      normalize.hpp/.cu       /255                (reused)
//
//  ----------------------------------------------------------- what this chain is
//  The classification task does NOT use a letterbox. Its preprocessing is the
//  torchvision/pillow chain produced by ultralytics' `classify_transforms(imgsz)`:
//
//      Resize(imgsz, BILINEAR)   shortest edge -> imgsz, longer edge proportional
//      CenterCrop(imgsz)
//      ToTensor()                HWC uint8 -> CHW float32, and / 255
//      Normalize(mean=0,std=1)   identity (ultralytics' DEFAULT_MEAN/DEFAULT_STD)
//
//  `Resize` receives a PIL image, so torchvision hands the geometry straight to
//  Pillow (`img.resize(size[::-1], interpolation)`). The resampling that runs is
//  therefore Pillow's libImaging -- which ANTIALIASES when downscaling and is
//  measured to differ from cv::resize(INTER_LINEAR) by up to 34/255 on 30-56% of
//  the pixels. That is why the cls chain cannot reuse the det chain's resize.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

namespace trt_alpha::kernels::ops {

// -----------------------------------------------------------------------------
//  PreprocessPilClsConfig -- the complete geometry of the cls preprocessing chain.
//
//  `Resize` produces newW x newH, `CenterCrop` then takes outW x outH starting at
//  (cropLeft, cropTop), and the tensor conversion only remaps channels and scale.
//  A pass whose output extent equals its input extent is skipped by Pillow, hence
//  needH / needV: when they are false that axis is a plain copy with NO filtering
//  and NO clip8 (this is the 640x640 in -> 640x640 out fast path).
// -----------------------------------------------------------------------------
struct PreprocessPilClsConfig
{
    int srcW = 0, srcH = 0;     //!< source image size (what setBatch delivers)
    int outW = 0, outH = 0;     //!< crop = network input size (square for cls)
    int newW = 0, newH = 0;     //!< size after Resize, before CenterCrop

    int cropLeft = 0, cropTop = 0;

    int needH = 0;              //!< newW != srcW  -> horizontal pass runs (0/1)
    int needV = 0;              //!< newH != srcH  -> vertical pass runs   (0/1)

    int ksizeH = 0, ksizeV = 0;

    //! Source column of crop output column `ox` (= ox + cropLeft).
    int resizedX(int ox) const noexcept { return ox + cropLeft; }
    //! Source row of crop output row `oy` (= oy + cropTop).  The horizontal pass
    //! is row-preserving, so resized row r is built from source row r.
    int resizedY(int oy) const noexcept { return oy + cropTop; }

    std::size_t outElems() const noexcept
    {
        return static_cast<std::size_t>(outW) * outH * 3;
    }

    //! False when the geometry cannot satisfy the network input: either a
    //! non-positive size, or a resize that came out SMALLER than the crop (which
    //! would need CenterCrop's zero padding -- not implemented).
    bool valid() const noexcept
    {
        return srcW > 0 && srcH > 0 && outW > 0 && outH > 0
            && newW >= outW && newH >= outH;
    }
};

// =============================================================================
//! Geometry of "Resize(shortest edge -> outW) then CenterCrop(outW x outH)".
//!
//! Composes the two individual operators' geometry helpers, so there is exactly
//! one implementation of each rule:
//!     computePilResizeShortestEdge (pil_resample.hpp)
//!     computeCenterCrop            (center_crop.hpp)
//! i.e. torchvision's `_compute_resized_output_size(size=int)` followed by
//! functional.center_crop's `int(round((image - crop) / 2.0))` -- note that is
//! Python's round(), i.e. half-to-even, and the resize truncates the long edge
//! with int().
//!
//! Never throws: an impossible geometry comes back with valid() == false, and the
//! plan constructors turn that into cudaErrorInvalidValue (numeric.hpp's error
//! convention, matching the other operators).
//!
//! Implemented in preprocess_fused_yolov26_cls.cu.
// =============================================================================
PreprocessPilClsConfig computePilClsGeometry(int srcW, int srcH, int outW, int outH);

}  // namespace trt_alpha::kernels::ops
