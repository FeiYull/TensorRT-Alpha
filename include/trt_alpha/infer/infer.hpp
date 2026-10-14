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
//    * async() -- asynchronous; the pipeline draws the results whenever draw /
//      show / save is on (draw is ON by default), so each Result carries both the
//      raw frame and the annotated copy
//    * every parameter is named in snake_case, matching the ini
// =============================================================================
#pragma once

#include "trt_alpha/cls/types.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/depth/types.hpp"
#include "trt_alpha/det/types.hpp"
#include "trt_alpha/kpt/types.hpp"
#include "trt_alpha/obb/types.hpp"
#include "trt_alpha/seg/types.hpp"
#include "trt_alpha/sem/types.hpp"

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
    yolov26,
    // YOLOV26 task variants (same backbone, different head). Each maps 1:1 to a
    // registration name and to a config in configs/. Append-only: the enum order
    // must stay in lock-step with kModelNames[] in infer.cpp.
    yolov26_cls,
    yolov26_depth,
    yolov26_obb,
    yolov26_pose,
    yolov26_seg,
    yolov26_sem,
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
//  SaveOption -- save switch + directory in one field (mirrors CLI `--save [dir]`).
//    p.save = true;        -> save to the default save/<net>/
//    p.save = "my_path";   -> save to my_path/
//    p.save = false;       -> do not save (default)
//  Accepts bool and string (literal or std::string). Source type is irrelevant:
//  image / video / camera / URL all follow the same rule.
// -----------------------------------------------------------------------------
struct SaveOption
{
    bool        enabled = false;   //!< whether to save at all
    std::string dir;               //!< empty = the default save/<net>/

    SaveOption() = default;
    SaveOption(bool b) : enabled(b) {}                                 // = true / false
    SaveOption(const char* d) : enabled(true), dir(d != nullptr ? d : "") {}
    SaveOption(const std::string& d) : enabled(true), dir(d) {}
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
//    * output:      show_window
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
    //! One string covering every input kind; the type is detected internally
    //! (datasource::resolveSource, the same rule as the CLI `--source`):
    //!   "data/bus.jpg" -> image,  "data/" -> image directory (one level),
    //!   "demo.mp4" -> local video,  "rtsp://..." -> stream,  "0" -> camera id.
    std::string source;
    int         camera_id = -1;     // explicit camera id (alternative to source="0")
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
    //! Save switch + directory: true -> save/<net>/, "path" -> path/.
    SaveOption  save;
    bool        show = false;
    std::string show_window;        // empty = use the ini
    //! Draw the results (boxes / masks / keypoints / ...) into an annotated copy,
    //! exposed as Frame::rendered(). ON by default: a plain run already hands back
    //! raw source + annotated copy + results. It does NOT write to disk or open a
    //! window -- that stays the job of save / show. Set it to false when you only
    //! want the data and want to skip the per-frame draw + copy entirely.
    bool        draw = true;

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

    //! The rendered (annotated) copy of this frame -- boxes / masks / keypoints /
    //! ... already drawn on. Filled only when a renderer ran (InferParams.draw /
    //! show / save); returns an empty view otherwise. `image()` above always stays
    //! the raw, untouched source, so a single run can hand back both.
    [[nodiscard]] core::BufferView rendered() const;
    [[nodiscard]] const std::vector<det::Detection>& boxes() const;
    [[nodiscard]] const std::vector<seg::Segmentation>& masks() const;
    [[nodiscard]] const std::vector<kpt::KeypointResult>& keypoints() const;
    [[nodiscard]] const std::vector<cls::ClassScore>& classifications() const;
    [[nodiscard]] const std::vector<obb::ObbDetection>& obbs() const;
    //! Dense depth map, float metres (NOT uint8). Empty when the model is not depth.
    [[nodiscard]] const depth::DepthMap& depth() const;
    //! Dense semantic map, uint8 class ids. Empty when the model is not semantic.
    [[nodiscard]] const sem::SemanticMask& semantic() const;

private:
    //! True when m_index is a real (non-padding) frame slot within `count` entries.
    //! A partial batch zero-fills the tail up to the engine batch size; those slots
    //! must not be handed back as if they were results.
    [[nodiscard]] bool validSlot(std::size_t count) const noexcept
    {
        if (m_result == nullptr || m_index >= count) {
            return false;
        }
        return m_result->validCount <= 0 ||
               m_index < static_cast<std::size_t>(m_result->validCount);
    }

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

    //! Number of *result* frames in this batch. A partial batch is zero-filled by
    //! the source up to the engine's batch size; those padding slots live in the
    //! buffer (views) but are NOT results -- frame iteration and the flattening
    //! accessors skip them. Falls back to the raw view count when validCount was
    //! never set (defensive: hand-built results in unit tests).
    [[nodiscard]] std::size_t size() const noexcept
    {
        const std::size_t total = m_result.views.size();
        const int valid = m_result.validCount;
        return (valid > 0 && static_cast<std::size_t>(valid) < total)
                   ? static_cast<std::size_t>(valid)
                   : total;
    }
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
    [[nodiscard]] std::vector<obb::ObbDetection> obbs() const;
    [[nodiscard]] std::vector<depth::DepthMap> depths() const;
    [[nodiscard]] std::vector<sem::SemanticMask> semantics() const;

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

    //! Fetch the next batch (blocking). false = end of stream.
    //! Available with or without a renderer: when show/save is set the pipeline
    //! renders first and hands the rendered result back here.
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
    //! Results are always available via Stream::get(): when show/save is set the
    //! pipeline renders first (and saves / shows) then hands the rendered result
    //! back; otherwise the caller gets the raw result.
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
