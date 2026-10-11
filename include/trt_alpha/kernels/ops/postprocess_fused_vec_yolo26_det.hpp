// =============================================================================
//  trt_alpha :: kernels :: ops :: postprocess_fused_vec_yolo26_det
// -----------------------------------------------------------------------------
//  The VECTORIZED variant of postprocess_fused_yolo26_det.
//
//  Same math, same order of operations, bit-identical output -- only the decode
//  phase changes:
//
//      scalar (postprocess_fused_yolo26_det)  this file (vec)
//      -------------------------------------  ---------------
//      1 thread = 1 anchor                    1 thread = 4 consecutive anchors
//      1 scalar load / channel / anchor       one 16B float4 load / channel
//      nc serial max/argmax per anchor        the 4 max/argmax reductions run
//                                             in parallel (independent lanes)
//
//  Why it is safe:
//      * y is channel-major: for a fixed channel the anchors are contiguous, so
//        4 consecutive anchors are 16B aligned whenever A is a multiple of 4
//        (8400 is). If A is not, the tail is handled by a scalar loop -- no
//        out-of-bounds access, no misalignment.
//      * Each lane still performs its own compare-and-select, so the chosen
//        class index and the resulting box are the same values as the scalar
//        version -- no "shared approximation" was introduced.
//      * The filter, the compaction, the sort and the greedy loop are literally
//        the same code (nms_utral_detail.cuh).
//
//  The config comes from yolo26_det_common.hpp, the same single definition the
//  scalar variant uses -- this header deliberately does NOT include the scalar
//  operator header: the two plans are independent, they only share parameters.
// =============================================================================
#pragma once

#include "trt_alpha/kernels/ops/yolo26_det_common.hpp"   // ProprocessFusedVecYolo26Config

namespace trt_alpha::kernels::ops {

// =============================================================================
//  CvProprocessFusedVecYolo26Plan -- interface identical to the scalar plan
// =============================================================================
class CvProprocessFusedVecYolo26Plan
{
public:
    explicit CvProprocessFusedVecYolo26Plan(const ProprocessFusedVecYolo26Config& cfg);

    ~CvProprocessFusedVecYolo26Plan();

    CvProprocessFusedVecYolo26Plan(const CvProprocessFusedVecYolo26Plan&)            = delete;
    CvProprocessFusedVecYolo26Plan& operator=(const CvProprocessFusedVecYolo26Plan&) = delete;
    CvProprocessFusedVecYolo26Plan(CvProprocessFusedVecYolo26Plan&& other) noexcept;

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    const ProprocessFusedVecYolo26Config& config() const noexcept { return cfg_; }

    int detStride() const noexcept { return cfg_.candStride(); }

    //! y : batch * (4+nc+extra) * anchors  ->  count/batch, dets/batch*maxDet*(6+extra)
    void launch(cudaStream_t stream, int batch,
                const float* y, int* count, float* dets) const;

private:
    void release() noexcept;

    ProprocessFusedVecYolo26Config cfg_;
    float*         d_buf_   = nullptr;
    int*           d_order_ = nullptr;
    unsigned char* d_alive_ = nullptr;
    int*           d_cnt_   = nullptr;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  proprocessFusedVecYolo26Device -- one-shot convenience version
// =============================================================================
void proprocessFusedVecYolo26Device(cudaStream_t stream, int batch,
                                    const float* y, int* count, float* dets,
                                    const ProprocessFusedVecYolo26Config& cfg);

}  // namespace trt_alpha::kernels::ops
