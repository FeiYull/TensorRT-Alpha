// =============================================================================
//  trt_alpha :: kernels :: ops :: yolov26_det_common
// -----------------------------------------------------------------------------
//  Types shared by the YOLOV26 detection postprocess operators.
//
//  The parameter struct lives HERE rather than in one of the operator headers,
//  so that
//      postprocess_fused_yolov26_det.hpp
//  and
//      postprocess_fused_vec_yolov26_det.hpp
//  stay two independent operators: neither has to include the other just to get
//  the parameters, and there is exactly ONE definition -- no copy to keep in
//  sync (the same reasoning that made computeLetterBox a single function).
// =============================================================================
#pragma once

namespace trt_alpha::kernels::ops {

// =============================================================================
//  ProprocessFusedYoloV26Config
// -----------------------------------------------------------------------------
//  Geometry and thresholds of "decode + conf filter + greedy NMS" for one image.
// =============================================================================
struct ProprocessFusedYoloV26Config
{
    int   anchors    = 8400;
    int   nc         = 80;
    int   extra      = 0;      //!< extra per-anchor channels (seg 32 / pose 51)
    int   maxBatch   = 8;
    int   maxDet     = 300;
    float confThresh = 0.25f;
    float iouThresh  = 0.70f;
    float maxWh      = 7680.0f;

    int candStride() const noexcept { return 6 + extra; }
};

//! The vectorized variant is parameter-compatible with the scalar one.
using ProprocessFusedVecYoloV26Config = ProprocessFusedYoloV26Config;

}  // namespace trt_alpha::kernels::ops
