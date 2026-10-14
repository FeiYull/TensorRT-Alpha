// =============================================================================
//  trt_alpha :: kernels :: ops :: topk_yolov26_cls
// -----------------------------------------------------------------------------
//  Top-k selection over a per-image score vector -- the WHOLE postprocess of the
//  YOLOV26 classification head.
//
//  Reference:
//      The exported yolo26n-cls.onnx output0 is [B, 1000] and has softmax
//      ALREADY baked in (verified with onnxruntime: min >= 0, sum == 1 exactly).
//      ultralytics' Results.probs then just takes torch.topk(probs, 5) --
//      there is no NMS, no decode, no coordinate work. So this operator is the
//      entire classification postprocess.
//
//  Determinism:
//      torch.topk documents no tie-break rule (its CPU implementation uses a
//      partial sort keyed on the value only). Exact ties among 1000 softmax
//      outputs do not occur in practice, but to stay reproducible this operator
//      always resolves a tie in favour of the SMALLER class index.
//
//  Conventions (same as every other operator in this directory):
//      * Plan = "an action", built once from immutable geometry.
//      * launch() issues exactly one kernel: no allocation, no copy, no sync.
//      * Nothing throws: a failure shows up as ok() == false / lastError().
//      * The output is [batch][k] pairs of (label, score), so the consumer never
//        has to guess the layout.
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {

// =============================================================================
//! Top-k of "scores -> (label, score) pairs".
// =============================================================================
class CvTopKPlan
{
public:
    //! numClass is the score-vector length, topK how many to keep.
    CvTopKPlan(int numClass, int topK, int maxBatch);

    bool matches(int numClass, int topK) const noexcept
    {
        return numClass_ == numClass && topK_ == topK;
    }

    [[nodiscard]] int numClass() const noexcept { return numClass_; }
    [[nodiscard]] int topK()     const noexcept { return topK_; }

    //! Floats per image in the destination: (label, score) * topK.
    [[nodiscard]] int outStride() const noexcept { return 2 * topK_; }

    //! Elements per image in the destination.
    [[nodiscard]] std::size_t outElems() const noexcept
    {
        return static_cast<std::size_t>(outStride());
    }

    [[nodiscard]] bool        ok()        const noexcept { return err_ == cudaSuccess; }
    [[nodiscard]] cudaError_t lastError() const noexcept { return err_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel. No allocation, no copy, no sync.
    //!   scores must hold batch * numClass floats
    //!   out    must hold batch * outStride() floats
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const float* scores, float* out) const;

private:
    int numClass_ = 0;
    int topK_     = 0;
    int maxBatch_ = 0;

    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//! topKDevice -- one-shot convenience version (no plan reuse).
// =============================================================================
void topKDevice(cudaStream_t stream, int batch,
                const float* scores, float* out,
                int numClass, int topK);

}  // namespace trt_alpha::kernels::ops
