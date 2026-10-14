// =============================================================================
//  trt_alpha :: kernels :: ops :: postprocess_fused_yolov26_obb
// -----------------------------------------------------------------------------
//  Fused "filter + conf sort + ROTATED NMS" for one image of a YOLOV26-OBB
//  head. The decode (dist2rbox + x stride + sigmoid) is already baked into the
//  exported graph, so nothing here touches the geometry -- it only selects.
//
//  Reference: ultralytics/utils/nms.py::non_max_suppression with rotated=True
//             (the OBB branch of DetectionPredictor.postprocess), which is
//             structurally different from the axis-aligned one:
//
//      boxes  = cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)   # xywhr
//      c      = x[:, 5:6] * max_wh                                  # class offset
//      i      = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
//
//  Two consequences, and both are the reason this is a separate operator:
//
//    1. The suppression rule is Fast-NMS (arXiv 1904.02689), NOT the greedy
//       loop the detector uses.  In code:
//
//           sorted_idx = argsort(scores, descending=True)
//           ious       = batch_probiou(boxes[sorted_idx], boxes[sorted_idx])
//           ious       = triu(ious, diagonal=1)            # strict upper
//           keep       = nonzero((ious >= iou_thres).sum(0) <= 0)
//           return sorted_idx[keep]
//
//       i.e. a box survives iff NO higher-scoring box overlaps it.  That is not
//       the same set the greedy loop produces (greedy also removes transitively
//       suppressed boxes), so reusing the detection NMS would be wrong, not
//       merely approximate.
//
//    2. The IoU is the probabilistic IoU of two Gaussians (batch_probiou), and
//       the class offset is added to BOTH centre coordinates -- so the class
//       separation still works out to "compare within a class".
//
//  Candidate row layout (candStride() = 6 + extra = 7):
//
//      [cx, cy, w, h, conf, cls, angle]
//
//  which is exactly the row ultralytics' NMS returns for rotated=True (the box
//  half is left in xywh because that branch skips xywh2xyxy, and the angle
//  rides along in the "extra" block).  So the consumer only has to map cx/cy/w/h
//  from network pixels to source pixels -- see the model class.
//
//  Naming: the sibling detection operator still spells it "Proprocess" because
//  a rename commit was scoped to file names only.  New code uses the correct
//  spelling rather than propagating the typo.
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/yolov26_det_common.hpp"   // ProprocessFusedYoloV26Config

namespace trt_alpha::kernels::ops {

// =============================================================================
//  CvPostprocessFusedYoloV26ObbPlan -- geometry fixed in the constructor
// =============================================================================
class CvPostprocessFusedYoloV26ObbPlan
{
public:
    explicit CvPostprocessFusedYoloV26ObbPlan(const ProprocessFusedYoloV26Config& cfg);

    ~CvPostprocessFusedYoloV26ObbPlan();

    CvPostprocessFusedYoloV26ObbPlan(const CvPostprocessFusedYoloV26ObbPlan&)            = delete;
    CvPostprocessFusedYoloV26ObbPlan& operator=(const CvPostprocessFusedYoloV26ObbPlan&) = delete;
    CvPostprocessFusedYoloV26ObbPlan(CvPostprocessFusedYoloV26ObbPlan&& other) noexcept;

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    const ProprocessFusedYoloV26Config& config() const noexcept { return cfg_; }

    //! Floats per candidate row: [cx, cy, w, h, conf, cls, angle...].
    int detStride() const noexcept { return cfg_.candStride(); }

    //! y : batch * (4+nc+extra) * anchors  ->  count/batch, dets/batch*maxDet*detStride()
    //! `extra` must be >= 1: the last channel is the angle.
    void launch(cudaStream_t stream, int batch,
                const float* y, int* count, float* dets) const;

private:
    void release() noexcept;

    ProprocessFusedYoloV26Config cfg_;
    float*         d_buf_   = nullptr;
    int*           d_order_ = nullptr;
    unsigned char* d_alive_ = nullptr;
    int*           d_cnt_   = nullptr;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  postprocessFusedYoloV26ObbDevice -- one-shot convenience version
// =============================================================================
void postprocessFusedYoloV26ObbDevice(cudaStream_t stream, int batch,
                                      const float* y, int* count, float* dets,
                                      const ProprocessFusedYoloV26Config& cfg);

}  // namespace trt_alpha::kernels::ops
