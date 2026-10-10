// =============================================================================
//  trt_alpha :: core :: model
// -----------------------------------------------------------------------------
//  IModel -- the unified top-level interface for every task
//  (detection / segmentation / classification / future tasks).
//
//  Design goals:
//    1. Task-agnostic: IModel only states "a batch of images goes in and the
//       results are fetched from here"; it says nothing about the output memory
//       layout. The concrete output structs are defined by the task base
//       classes (det::IDetector / seg::ISegmentor / cls::IClassifier).
//    2. Batch-oriented: setBatch takes a batch of same-sized images (Batch).
//    3. Adding a model = 1 .cpp + 1 line of TRT_ALPHA_REGISTER_MODEL, with no
//       invasive changes elsewhere.
//
//  Call-sequence contract (the only legal order):
//    init(cfg) -> [ setBatch(batch) -> preprocess() -> infer() -> postprocess()
//                   -> commitResult(result) -> reset() ] loop
//    postprocess() returning means this round's GPU results are ready
//    (it performs D2H + stream sync internally).
//
//  Error-handling contract:
//    Init / argument / device errors always throw std::runtime_error (with
//    context). init() must validate ModelConfig itself (e.g. whether numClass
//    matches the engine's actual nc).
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_config.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace trt_alpha {

//! Unified top-level interface for every task.
class IModel
{
public:
    virtual ~IModel() = default;

    IModel(const IModel&) = delete;
    IModel& operator=(const IModel&) = delete;

    //! Registration name (must match the `model` field in the INI file).
    [[nodiscard]] virtual const std::string& name() const noexcept = 0;

    //! Initialize: load the engine and allocate device memory. Throws on failure.
    //! Must validate ModelConfig internally (e.g. numClass vs. the engine's nc).
    virtual void init(const core::ModelConfig& cfg) = 0;

    //! Upload one batch of images to device memory. Buffers in a Batch are contiguous.
    virtual void setBatch(const core::Batch& batch) = 0;

    //! CUDA preprocessing (letterbox / normalization / HWC->NCHW, etc.).
    //! Implemented by each model.
    virtual void preprocess() = 0;

    //! Asynchronous inference through enqueueV3.
    virtual void infer() = 0;

    //! Decode + NMS + D2H; results are ready once this returns.
    //! Implemented by each model.
    virtual void postprocess() = 0;

    //! Move this round's results into `out` (commit semantics).
    virtual void commitResult(core::BatchResult& out) = 0;

    //! Clear this round's state.
    virtual void reset() = 0;

    //! Engine I/O tensor descriptions (for debugging).
    [[nodiscard]] virtual const std::vector<core::TensorDesc>& describe() const noexcept = 0;

    //! The config actually in effect after init() (batch / dstH / dstW as
    //! resolved from the engine). May differ from the cfg passed to init():
    //! static shapes are corrected to the values declared by the engine.
    [[nodiscard]] virtual const core::ModelConfig& config() const noexcept = 0;

protected:
    IModel() = default;
};

//! Batch-capacity guard for setBatch -- shared by all tasks (det / seg / kpt / cls).
//!
//! Why it is mandatory: every staging / output buffer of a model is sized for
//! the engine-resolved batch, while the Batch handed to setBatch comes from the
//! caller. When views outnumber that capacity, preprocess / decode drive their
//! kernels by views.size() and write past the allocated memory (undefined
//! behaviour, and silently so). So this throws outright, following the rule
//! "config disagrees with engine capability -> fail explicitly": it neither
//! truncates nor stays silent.
//!
//! This is the base-class funnel: any new model that takes its batch through
//! this function inherits the guard (compare InferencePool::submit -- the
//! production path carries an equivalent check even further upstream).
//!
//! @param model the model itself (used to read the config in effect after init())
//! @param batch this batch's input
//! @param who   model name (goes into the log / exception message)
//! @return the actual batch size (== batch.views.size()), so callers can assign in one line
inline int requireBatchCapacity(const IModel& model, const core::Batch& batch,
                                const char* who)
{
    const int capacity = model.config().batchSize;
    const int n = static_cast<int>(batch.views.size());
    if (capacity > 0 && n > capacity)
    {
        const std::string msg =
            std::string(who) + ": batch of " + std::to_string(n) +
            " images exceeds model capacity " + std::to_string(capacity) +
            " (buffers are sized by the engine-resolved batch; raise [input]"
            " batch_size within the engine profile instead)";
        TRT_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }
    return n;
}

}  // namespace trt_alpha
