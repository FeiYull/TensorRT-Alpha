// =============================================================================
//  trt_alpha :: infer
// -----------------------------------------------------------------------------
//  Infer -- the high-level "load a model + run inference" interface.
//
//  Usage (synchronous):
//      trt_alpha::InferParams p;
//      p.model_type  = trt_alpha::ModelType::yolov8;
//      p.config_path = "configs/yolov8.ini";
//      p.source      = "data/bus.jpg";
//      p.conf_thresh = 0.5f;
//
//      trt_alpha::Infer model(p);
//      auto r = model.run();
//      for (auto& f : r.frames()) {
//          auto img = f.image();
//          for (auto& b : f.boxes()) { ... }
//      }
//
//  Usage (asynchronous):
//      p.source = "data/people.mp4";
//      trt_alpha::Infer model(p);
//      auto stream = model.async();
//      trt_alpha::Result r;
//      while (stream.get(r)) {
//          for (auto& f : r.frames()) { ... }
//      }
//
//  Design:
//    * one InferParams struct holds every parameter
//    * on construction: read the ini (base + special merged), apply the p
//      overrides, print, then build the pool
//    * run()   -- synchronous, never renders, returns one batch of Result
//    * async() -- asynchronous; when show || save is set, Pipeline renders by
//      itself, otherwise the caller gets Result
//    * every parameter is named in snake_case, matching the ini
// =============================================================================
#pragma once

#include "trt_alpha/cls/types.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/det/types.hpp"
#include "trt_alpha/kpt/types.hpp"
#include "trt_alpha/seg/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_alpha {

// -----------------------------------------------------------------------------
//  ModelType -- an enum of registered models (one-to-one with the registration
//  names in ModelRegistry).
// -----------------------------------------------------------------------------
enum class ModelType
{
    yolov3,
    yolov4,
    yolov5,
    yolov6,
    yolov7,
    yolov8,
    yolov8_pose,
    yolov8_seg,
    yolox,
    yolor,
    yolo_nas,
    efficientdet,
    yunet,
    u2net,
};

// -----------------------------------------------------------------------------
//  QueuePolicy -- the policy when the queue is full. Unset = no override (use
//  the ini default).
// -----------------------------------------------------------------------------
enum class QueuePolicy
{
    Unset,
    DropOldest,
    DropNewest,
    Block,
};

// -----------------------------------------------------------------------------
//  InferParams -- the Infer constructor parameters.
//
//  Required:
//    * model_type  -- the model type
//    * config_path -- the ini path (e.g. "configs/yolov8.ini")
//    * a source: source (image / video / URL) or camera_id (camera), one of them
//
//  Optional (empty / -1 / Unset = no override, use the ini):
//    * load time:   engine / class_names_file / batch_size / dst_h / dst_w
//    * run time:    conf_thresh / iou_thresh / top_k
//    * concurrency: workers / max_queue_size / result_queue_size / policy
//    * output:      save_dir / show_window
//
//  InferParams only (not present in the ini):
//    * save / show / loop
// -----------------------------------------------------------------------------
struct InferParams
{
    // ---- Required ----
    ModelType   model_type{};
    std::string config_path;

    // ---- Source (source or camera_id, one of them) ----
    std::string source;             // image / video / RTSP / HTTP URL
    int         camera_id = -1;
    bool        loop = false;       // loop the video (only when source is a video)

    // ---- Override the ini (load time) ----
    std::string engine;             // empty = use the ini
    std::string class_names_file;   // empty = use the ini
    int         batch_size = -1;
    int         dst_h = -1;
    int         dst_w = -1;

    // ---- Override the ini (run time) ----
    float conf_thresh = -1.f;
    float iou_thresh  = -1.f;
    int   top_k       = -1;

    // ---- Output (InferParams only) ----
    bool        save = false;
    std::string save_dir;           // empty = use the ini
    bool        show = false;
    std::string show_window;        // empty = use the ini

    // ---- Override the ini (concurrency) ----
    int         workers = -1;
    int         max_queue_size = -1;
    int         result_queue_size = -1;
    QueuePolicy policy = QueuePolicy::Unset;

    // ---- Catch-all: any ini field ----
    std::unordered_map<std::string, std::string> extras;

    //! Validate: model_type is legal / config_path is non-empty / exactly one source.
    void validate() const;
};

// -----------------------------------------------------------------------------
//  Frame -- one frame within a batch (a read-only view living as long as the
//  Result).
// -----------------------------------------------------------------------------
class Frame
{
public:
    Frame() = default;
    Frame(const core::BatchResult* result, std::size_t index)
        : m_result(result)
        , m_index(index)
    {
    }

    [[nodiscard]] core::BufferView image() const;
    [[nodiscard]] const std::vector<det::Detection>& boxes() const;
    [[nodiscard]] const std::vector<seg::Segmentation>& masks() const;
    [[nodiscard]] const std::vector<kpt::KeypointResult>& keypoints() const;
    [[nodiscard]] const std::vector<cls::ClassScore>& classifications() const;

private:
    const core::BatchResult* m_result = nullptr;
    std::size_t m_index = 0;
};

// -----------------------------------------------------------------------------
//  Result -- one batch of inference results.
// -----------------------------------------------------------------------------
class Result
{
public:
    Result() = default;
    explicit Result(core::BatchResult&& r) : m_result(std::move(r)) {}

    [[nodiscard]] std::size_t size() const noexcept { return m_result.views.size(); }
    [[nodiscard]] int valid_count() const noexcept { return m_result.validCount; }
    [[nodiscard]] int source_id() const noexcept { return m_result.sourceId; }
    [[nodiscard]] std::uint64_t first_frame_index() const noexcept
    {
        return m_result.firstFrameIndex;
    }
    [[nodiscard]] double inference_ms() const noexcept { return m_result.inferenceMs; }
    [[nodiscard]] bool empty() const noexcept { return m_result.empty(); }

    [[nodiscard]] Frame operator[](std::size_t i) const { return Frame(&m_result, i); }
    [[nodiscard]] std::vector<Frame> frames() const;

    //! Flattening convenience accessors (all frames concatenated).
    [[nodiscard]] std::vector<det::Detection> boxes() const;
    [[nodiscard]] std::vector<seg::Segmentation> masks() const;
    [[nodiscard]] std::vector<kpt::KeypointResult> keypoints() const;
    [[nodiscard]] std::vector<cls::ClassScore> classifications() const;

private:
    core::BatchResult m_result;
};

// -----------------------------------------------------------------------------
//  Stream -- the return value of async(). Results are fetched batch by batch.
// -----------------------------------------------------------------------------
class Stream
{
public:
    Stream();
    ~Stream();

    Stream(Stream&&) noexcept;
    Stream& operator=(Stream&&) noexcept;

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    //! Fetch the next batch (blocking). false = end of stream, or render mode
    //! (which produces no Result).
    bool get(Result& out);

    //! Stop explicitly (thread-safe, idempotent).
    void stop();

    [[nodiscard]] bool running() const noexcept;

private:
    friend class Infer;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// -----------------------------------------------------------------------------
//  Infer -- the high-level API.
//
//  Lifetime:
//    * construction: merge the config + build the pool (engine load + device
//      memory allocation)
//    * run() / async(): run inference
//    * destruction: stop the pool
// -----------------------------------------------------------------------------
class Infer
{
public:
    explicit Infer(const InferParams& p);
    ~Infer();

    Infer(const Infer&) = delete;
    Infer& operator=(const Infer&) = delete;
    Infer(Infer&&) noexcept;
    Infer& operator=(Infer&&) noexcept;

    //! Asynchronous: returns a Stream.
    //! When show || save is set, Pipeline starts a render thread and consumes
    //! results itself (Stream::get() returns false). Otherwise the caller uses
    //! Stream::get() to fetch Result.
    [[nodiscard]] Stream async();

private:
    struct Impl;
    //! Shared ownership: the Stream returned by async() borrows the pool /
    //! renderer inside Impl (Pipeline references them by raw pointer), so the
    //! Stream must keep Impl alive -- otherwise with Infer destroyed first while
    //! the Stream is still in use we would have a dangling reference (UB).
    std::shared_ptr<Impl> m_impl;
};

}  // namespace trt_alpha
