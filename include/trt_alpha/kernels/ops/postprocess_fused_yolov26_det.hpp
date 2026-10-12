// =============================================================================
//  trt_alpha :: kernels :: ops :: postprocess_fused_yolov26_det
// -----------------------------------------------------------------------------
//  The FUSED variant of "decode_yolov26 + nms_utral": one kernel does everything.
//
//      y (batch, 4+nc+extra, A)
//          -> decode each anchor (max/argmax over classes, xywh -> xyxy)
//          -> drop conf <= confThresh
//          -> compact into a per-image candidate buffer
//          -> stable sort by conf, descending
//          -> greedy NMS (same code as nms_utral, see nms_utral_detail.cuh)
//          -> (count[], dets[])
//
//  Compared with running the two operators back to back:
//      * 1 kernel launch instead of 3
//      * the full (batch, A, 6+extra) candidate tensor is never written to and
//        read back from global memory -- only the survivors are materialised
//      * the sort and the greedy loop are the exact same device code, so the
//        result cannot drift from the unfused path
//
//  The result is bit-identical to decode_yolov26 followed by nms_utral, and hence
//  to the Python reference. That is the invariant the demo checks.
//
//  Output: count : int   [batch]
//          dets  : float [batch][maxDet][6+extra]   (input coordinates)
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/yolov26_det_common.hpp"   // ProprocessFusedYoloV26Config

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {

// =============================================================================
//  CvProprocessFusedYoloV26Plan
// =============================================================================
class CvProprocessFusedYoloV26Plan
{
public:
    explicit CvProprocessFusedYoloV26Plan(const ProprocessFusedYoloV26Config& cfg);

    ~CvProprocessFusedYoloV26Plan();

    CvProprocessFusedYoloV26Plan(const CvProprocessFusedYoloV26Plan&)            = delete;
    CvProprocessFusedYoloV26Plan& operator=(const CvProprocessFusedYoloV26Plan&) = delete;
    CvProprocessFusedYoloV26Plan(CvProprocessFusedYoloV26Plan&& other) noexcept;

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    const ProprocessFusedYoloV26Config& config() const noexcept { return cfg_; }

    int detStride() const noexcept { return cfg_.candStride(); }

    // -------------------------------------------------------------------------
    //! Hot path: one kernel launch, no allocation / copy / sync.
    //!   y     : batch * (4+nc+extra) * anchors
    //!   count : batch ints                    (written)
    //!   dets  : batch * maxDet * (6+extra)    (written)
    // -------------------------------------------------------------------------
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
//  proprocessFusedYoloV26Device -- one-shot convenience version
// =============================================================================
void proprocessFusedYoloV26Device(cudaStream_t stream, int batch,
                                 const float* y, int* count, float* dets,
                                 const ProprocessFusedYoloV26Config& cfg);

}  // namespace trt_alpha::kernels::ops
