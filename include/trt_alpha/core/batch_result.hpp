// =============================================================================
//  trt_alpha :: core :: batch_result
// -----------------------------------------------------------------------------
//  BatchResult -- the inference results for one batch of images.
//
//  Fields:
//    * sourceId / firstFrameIndex: mirrored from Batch (source disambiguation
//      + frame index)
//    * buffer / views / validCount: the source images (the very batch fed into
//      inference)
//    * detections / segmentations / classifications / keypoints: per-task
//      results (filled according to the model type; task fields that were not
//      produced stay empty)
//    * inferenceMs: end-to-end time for this batch (performance metadata)
//
//  Lifetime:
//    * buffer is a shared_ptr (keeps views[i].data valid)
//    * Each task's results own their data (value semantics inside the vectors)
//    * The render stage reads the source image via views[i] and the results via
//      detections[i] and friends
// =============================================================================
#pragma once

#include "trt_alpha/cls/types.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/det/types.hpp"
#include "trt_alpha/seg/types.hpp"
#include "trt_alpha/kpt/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::core {

//! Inference results for one batch of images.
struct BatchResult
{
    // ---- Source information (mirrors Batch) ----
    int sourceId = -1;                           //!< source identifier
    std::uint64_t firstFrameIndex = 0;           //!< frame index of this batch's first frame

    // ---- Source images (the batch that was fed in) ----
    std::shared_ptr<Buffer> buffer;              //!< owner of the source images (one contiguous block)
    std::vector<BufferView> views;               //!< view of each image
    int validCount = 0;                          //!< number of valid frames (<= views.size())

    //! Per-frame "source stem" (extension / index stripped); length == validCount.
    //! Files are named after it when saved (image source = original filename);
    //! when empty the renderer falls back to frame_<index>.
    std::vector<std::string> frameNames;

    // ---- Per-task results (filled by model type; unmatched tasks stay empty) ----
    std::vector<std::vector<det::Detection>> detections;          //!< detections per image
    std::vector<std::vector<seg::Segmentation>> segmentations;    //!< segmentations per image
    std::vector<std::vector<cls::ClassScore>> classifications;    //!< classifications per image
    std::vector<std::vector<kpt::KeypointResult>> keypoints;      //!< pose results per image

    // ---- Performance metadata ----
    //! End-to-end time for this batch (ms): the whole
    //! setBatch -> preprocess -> infer -> postprocess path.
    //! Note this is not pure GPU inference time -- infer() only enqueues, so the
    //! actual GPU work is counted inside this total.
    double inferenceMs = 0.0;

    //! Whether the result is empty.
    [[nodiscard]] bool empty() const noexcept
    {
        return buffer == nullptr || views.empty();
    }

    //! Batch size.
    [[nodiscard]] std::size_t size() const noexcept { return views.size(); }
};

}  // namespace trt_alpha::core
