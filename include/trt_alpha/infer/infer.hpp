// =============================================================================
//  trt_alpha :: infer
// -----------------------------------------------------------------------------
//  Infer —— 高层"加载模型 + 跑推理"接口。
//
//  用法（同步）：
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
//  用法（异步）：
//      p.source = "data/people.mp4";
//      trt_alpha::Infer model(p);
//      auto stream = model.async();
//      trt_alpha::Result r;
//      while (stream.get(r)) {
//          for (auto& f : r.frames()) { ... }
//      }
//
//  设计：
//    * 一个结构体 InferParams 装所有参数
//    * 构造时：读 ini（base + special 合并）→ 应用 p 覆盖 → 打印 → 建 pool
//    * run()  —— 同步，永远不渲染，返回一批 Result
//    * async()—— 异步，show || save 时 Pipeline 自己渲染，否则用户拿 Result
//    * 全部参数命名用 snake_case，和 ini 对齐
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
//  ModelType —— 已注册模型的枚举（和 ModelRegistry 的注册名一一对应）。
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
//  QueuePolicy —— 队列满时策略。Unset = 不覆盖（走 ini 默认）。
// -----------------------------------------------------------------------------
enum class QueuePolicy
{
    Unset,
    DropOldest,
    DropNewest,
    Block,
};

// -----------------------------------------------------------------------------
//  InferParams —— Infer 构造参数。
//
//  必填：
//    * model_type  —— 模型类型
//    * config_path —— ini 路径（如 "configs/yolov8.ini"）
//    * 源：source（图片 / 视频 / URL）或 camera_id（摄像头），二选一
//
//  可选（空 / -1 / Unset = 不覆盖，走 ini）：
//    * 加载期：engine / class_names_file / batch_size / dst_h / dst_w
//    * 运行期：conf_thresh / iou_thresh / top_k
//    * 并发：  workers / max_queue_size / result_queue_size / policy
//    * 输出：  save_dir / show_window
//
//  仅 InferParams（ini 里没有）：
//    * save / show / loop
// -----------------------------------------------------------------------------
struct InferParams
{
    // ---- 必填 ----
    ModelType   model_type{};
    std::string config_path;

    // ---- 源（source 或 camera_id 二选一）----
    std::string source;             // 图片 / 视频 / RTSP / HTTP URL
    int         camera_id = -1;
    bool        loop = false;       // 视频循环（仅 source 是视频时有效）

    // ---- 覆盖 ini（加载期）----
    std::string engine;             // 空 = 用 ini
    std::string class_names_file;   // 空 = 用 ini
    int         batch_size = -1;
    int         dst_h = -1;
    int         dst_w = -1;

    // ---- 覆盖 ini（运行期）----
    float conf_thresh = -1.f;
    float iou_thresh  = -1.f;
    int   top_k       = -1;

    // ---- 输出（仅 InferParams）----
    bool        save = false;
    std::string save_dir;           // 空 = 用 ini
    bool        show = false;
    std::string show_window;        // 空 = 用 ini

    // ---- 覆盖 ini（并发）----
    int         workers = -1;
    int         max_queue_size = -1;
    int         result_queue_size = -1;
    QueuePolicy policy = QueuePolicy::Unset;

    // ---- 兜底：任意 ini 字段 ----
    std::unordered_map<std::string, std::string> extras;

    //! 校验：model_type 合法 / config_path 非空 / 源二选一。
    void validate() const;
};

// -----------------------------------------------------------------------------
//  Frame —— 一批里的一帧（只读视图，生命周期跟 Result 一致）。
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
//  Result —— 一批推理结果。
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

    //! 拍平便利访问器（所有帧拼一起）。
    [[nodiscard]] std::vector<det::Detection> boxes() const;
    [[nodiscard]] std::vector<seg::Segmentation> masks() const;
    [[nodiscard]] std::vector<kpt::KeypointResult> keypoints() const;
    [[nodiscard]] std::vector<cls::ClassScore> classifications() const;

private:
    core::BatchResult m_result;
};

// -----------------------------------------------------------------------------
//  Stream —— async() 的返回值。按 batch 拿结果。
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

    //! 拿下一批（阻塞）。false = 流结束 / 渲染模式（不产 Result）。
    bool get(Result& out);

    //! 主动停止（线程安全，幂等）。
    void stop();

    [[nodiscard]] bool running() const noexcept;

private:
    friend class Infer;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// -----------------------------------------------------------------------------
//  Infer —— 高层 API。
//
//  生命周期：
//    * 构造：合并配置 + 建 pool（engine 加载 + 显存分配）
//    * run() / async()：跑推理
//    * 析构：停 pool
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

    //! 异步：返回 Stream。
    //! show || save 时，Pipeline 起渲染线程自己消费（Stream::get() 返回 false）。
    //! 否则，用户用 Stream::get() 拿 Result。
    [[nodiscard]] Stream async();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace trt_alpha