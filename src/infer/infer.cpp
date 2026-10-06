// =============================================================================
//  trt_alpha :: infer（实现）
// =============================================================================
#include "trt_alpha/infer/infer.hpp"

#include "trt_alpha/core/config.hpp"
#include "trt_alpha/core/inference_pool.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/core/paths.hpp"
#include "trt_alpha/datasource/opencv_source.hpp"
#include "trt_alpha/pipeline/pipeline.hpp"
#include "trt_alpha/pipeline/pipeline_config.hpp"
#include "trt_alpha/renderer/opencv_renderer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_alpha {

// =============================================================================
//  ModelType -> 注册名
// =============================================================================
namespace {

const char* kModelNames[] = {
    "yolov3", "yolov4", "yolov5", "yolov6", "yolov7", "yolov8",
    "yolov8_pose", "yolov8_seg",
    "yolox", "yolor", "yolo_nas",
    "efficientdet", "yunet", "u2net",
};

const char* registry_name(ModelType t)
{
    const int idx = static_cast<int>(t);
    const int total = static_cast<int>(sizeof(kModelNames) / sizeof(kModelNames[0]));
    if (idx < 0 || idx >= total) {
        throw std::runtime_error("Infer: invalid ModelType");
    }
    return kModelNames[idx];
}

core::BoundedQueue<std::future<core::BatchResult>>::FullPolicy
to_full_policy(QueuePolicy p)
{
    using FP = core::BoundedQueue<std::future<core::BatchResult>>::FullPolicy;
    switch (p) {
    case QueuePolicy::DropNewest: return FP::DropNewest;
    case QueuePolicy::Block:      return FP::Block;
    case QueuePolicy::DropOldest:
    case QueuePolicy::Unset:
    default:                      return FP::DropOldest;
    }
}

//! 是否为视频文件后缀（小写比较；先剥离 URL 的查询串/锚点）。
//! 按后缀而非子串判定：目录名 "a.mp4_backup" 里的图不该被当成视频。
bool hasVideoExt(const std::string& src)
{
    const std::string path = src.substr(0, src.find_first_of("?#"));
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos)
    {
        return false;
    }
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    static const std::vector<std::string> kExts = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
        ".ts", ".flv", ".mpg", ".mpeg", ".wmv",
    };
    return std::find(kExts.begin(), kExts.end(), ext) != kExts.end();
}

//! 构造数据源（根据 source / camera_id 推断类型）。
datasource::SourceConfig build_source_config(const InferParams& p, int batch_size)
{
    datasource::SourceConfig cfg;
    cfg.batchSize = batch_size;
    cfg.sourceId = 0;
    cfg.loop = p.loop;

    if (!p.source.empty())
    {
        // 判断源类型：URL（流）/ 视频文件 / 图片（也可能是目录，openImage 内部再降级）。
        // URL 一律当流：cv::imread 只认本地文件，远程资源只能走 FFmpeg 后端。
        const std::string& s = p.source;
        cfg.path = s;
        cfg.type = (core::Paths::isUrl(s) || hasVideoExt(s))
                       ? datasource::SourceType::Video
                       : datasource::SourceType::Image;
    }
    else if (p.camera_id >= 0)
    {
        cfg.type = datasource::SourceType::Camera;
        cfg.cameraId = p.camera_id;
    }
    else
    {
        throw std::runtime_error("InferParams: no source (source / camera_id)");
    }
    return cfg;
}

}  // namespace

// =============================================================================
//  InferParams::validate
// =============================================================================
void InferParams::validate() const
{
    if (config_path.empty()) {
        throw std::runtime_error("InferParams: config_path is empty");
    }
    if (source.empty() && camera_id < 0) {
        throw std::runtime_error("InferParams: no source (source / camera_id)");
    }
    if (!source.empty() && camera_id >= 0) {
        throw std::runtime_error(
            "InferParams: both source and camera_id are set (choose one)");
    }
}

// =============================================================================
//  Frame
// =============================================================================
core::BufferView Frame::image() const
{
    if (m_result == nullptr || m_index >= m_result->views.size()) {
        return {};
    }
    return m_result->views[m_index];
}

const std::vector<det::Detection>& Frame::boxes() const
{
    static const std::vector<det::Detection> kEmpty;
    if (m_result == nullptr || m_index >= m_result->detections.size()) {
        return kEmpty;
    }
    return m_result->detections[m_index];
}

const std::vector<seg::Segmentation>& Frame::masks() const
{
    static const std::vector<seg::Segmentation> kEmpty;
    if (m_result == nullptr || m_index >= m_result->segmentations.size()) {
        return kEmpty;
    }
    return m_result->segmentations[m_index];
}

const std::vector<kpt::KeypointResult>& Frame::keypoints() const
{
    static const std::vector<kpt::KeypointResult> kEmpty;
    if (m_result == nullptr || m_index >= m_result->keypoints.size()) {
        return kEmpty;
    }
    return m_result->keypoints[m_index];
}

const std::vector<cls::ClassScore>& Frame::classifications() const
{
    static const std::vector<cls::ClassScore> kEmpty;
    if (m_result == nullptr || m_index >= m_result->classifications.size()) {
        return kEmpty;
    }
    return m_result->classifications[m_index];
}

// =============================================================================
//  Result
// =============================================================================
std::vector<Frame> Result::frames() const
{
    std::vector<Frame> out;
    out.reserve(m_result.views.size());
    for (std::size_t i = 0; i < m_result.views.size(); ++i) {
        out.emplace_back(&m_result, i);
    }
    return out;
}

std::vector<det::Detection> Result::boxes() const
{
    std::vector<det::Detection> out;
    for (const auto& v : m_result.detections) {
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<seg::Segmentation> Result::masks() const
{
    std::vector<seg::Segmentation> out;
    for (const auto& v : m_result.segmentations) {
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<kpt::KeypointResult> Result::keypoints() const
{
    std::vector<kpt::KeypointResult> out;
    for (const auto& v : m_result.keypoints) {
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<cls::ClassScore> Result::classifications() const
{
    std::vector<cls::ClassScore> out;
    for (const auto& v : m_result.classifications) {
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

// =============================================================================
//  Infer::Impl
// =============================================================================
struct Infer::Impl
{
    InferParams params;
    core::ModelConfig final_cfg;
    std::shared_ptr<core::InferencePool> pool;
    renderer::OpenCVRenderer renderer;

    core::ModelConfig merge_config() const;
    void ensure_pool();
};
core::ModelConfig Infer::Impl::merge_config() const
{
    core::ModelConfig cfg = core::loadModelConfig(params.config_path);

    // 应用 InferParams 覆盖（非默认值才覆盖）
    if (!params.engine.empty())          cfg.engine = params.engine;
    if (!params.class_names_file.empty()) cfg.classNamesFile = params.class_names_file;
    if (params.batch_size > 0)           cfg.batchSize = params.batch_size;
    if (params.dst_h > 0)                cfg.dstH = params.dst_h;
    if (params.dst_w > 0)                cfg.dstW = params.dst_w;

    if (params.conf_thresh >= 0.f) cfg.extras["conf_thresh"] = std::to_string(params.conf_thresh);
    if (params.iou_thresh  >= 0.f) cfg.extras["iou_thresh"]  = std::to_string(params.iou_thresh);
    if (params.top_k > 0)          cfg.extras["top_k"]       = std::to_string(params.top_k);

    if (params.workers > 0)           cfg.extras["workers"] = std::to_string(params.workers);
    if (params.max_queue_size > 0)    cfg.extras["max_queue_size"] = std::to_string(params.max_queue_size);
    if (params.result_queue_size > 0) cfg.extras["result_queue_size"] = std::to_string(params.result_queue_size);

    if (!params.save_dir.empty())    cfg.extras["save_dir"] = params.save_dir;
    if (!params.show_window.empty()) cfg.extras["show_window"] = params.show_window;

    for (const auto& [k, v] : params.extras) {
        cfg.extras[k] = v;
    }

    cfg.classNames = core::loadClassNamesFile(cfg.classNamesFile);
    return cfg;
}

void Infer::Impl::ensure_pool()
{
    if (pool != nullptr) {
        return;
    }

    const std::size_t workers =
        (params.workers > 0) ? static_cast<std::size_t>(params.workers) : std::size_t{0};

    // 队列大小优先级：InferParams > base.ini > 16
    int max_q_int = params.max_queue_size;
    if (max_q_int <= 0) {
        max_q_int = final_cfg.getInt("max_queue_size", 16);
    }
    const std::size_t max_q = static_cast<std::size_t>(max_q_int > 0 ? max_q_int : 16);

    const std::string net = registry_name(params.model_type);

    pool = std::make_shared<core::InferencePool>(
        final_cfg,
        [net]() -> std::unique_ptr<IModel> {
            return ModelRegistry::instance().create(net);
        },
        workers,
        max_q);

    // 池已按引擎实际能力修正 batch（静态固定 / 动态钳制）
    if (pool->resolvedBatch() != final_cfg.batchSize)
    {
        TRT_LOG_INFO("Infer: batch corrected by engine: config "
                     << final_cfg.batchSize << " -> " << pool->resolvedBatch());
    }
    TRT_LOG_INFO("Infer: effective batch = " << pool->resolvedBatch());

    // 本次实际生效的配置（与 CLI run 命令同一套展示口径）。
    // 放在池建好之后：此时模型已 init（引擎真相可得），且
    // 模型 init 期间的读取痕迹都落在 pool->modelConfig() 上，
    // 因此 ini 里的死键能被正确标成 [unused]。
    core::logConfigBox(pool->modelConfig(), net, params.config_path,
                       &pool->ioDesc(), pool->resolvedBatch());
}

// =============================================================================
//  Infer
// =============================================================================
Infer::Infer(const InferParams& p)
    : m_impl(std::make_shared<Impl>())
{
    p.validate();
    m_impl->params = p;
    m_impl->final_cfg = m_impl->merge_config();
    // 配置展示推迟到 ensure_pool()（首次 run/async）：那时才有引擎真相，
    // 且模型 init 的"读取痕迹"才完整 —— 目的是只打一次、且打的是真生效的口径。
}

Infer::~Infer() = default;
Infer::Infer(Infer&&) noexcept = default;
Infer& Infer::operator=(Infer&&) noexcept = default;

// =============================================================================
//  Stream::Impl
// =============================================================================
struct Stream::Impl
{
    std::unique_ptr<pipeline::Pipeline> pipeline;
    //! 保活 Infer::Impl：Pipeline 持有 pool / renderer 的**裸指针**，
    //! 它们都是 Infer::Impl 的成员（或成员持有的），所以 Stream 只要
    //! 抓住 Impl 的强引用，就能保证"Stream 活着 → 这些资源活着"。
    std::shared_ptr<void> owner;
};

Stream::Stream() = default;
Stream::~Stream() { stop(); }
Stream::Stream(Stream&&) noexcept = default;
Stream& Stream::operator=(Stream&&) noexcept = default;

bool Stream::get(Result& out)
{
    if (m_impl == nullptr || m_impl->pipeline == nullptr) {
        return false;
    }

    core::BatchResult br;
    bool got = false;
    if (m_impl->pipeline->hasRenderer()) {
        got = m_impl->pipeline->popProcessed(br);   // 渲染后拿
    } else {
        got = m_impl->pipeline->popResult(br);      // 非渲染，原始结果
    }

    if (!got) {
        return false;
    }
    out = Result(std::move(br));
    return true;
}

void Stream::stop()
{
    if (m_impl == nullptr || m_impl->pipeline == nullptr) {
        return;
    }
    if (m_impl->pipeline->running()) {
        m_impl->pipeline->stop();
        m_impl->pipeline->waitForCompletion();
    }
    m_impl->pipeline.reset();
}

bool Stream::running() const noexcept
{
    return m_impl != nullptr &&
           m_impl->pipeline != nullptr &&
           m_impl->pipeline->running();
}

// =============================================================================
//  Infer::async
// =============================================================================
Stream Infer::async()
{
    m_impl->ensure_pool();

    const bool render = m_impl->params.show || m_impl->params.save;

    pipeline::PipelineConfig pcfg;
    pcfg.sources.push_back(
        std::make_unique<datasource::OpenCVSource>(
            build_source_config(m_impl->params, m_impl->pool->resolvedBatch())));
    pcfg.pools = { m_impl->pool.get() };
    pcfg.sourceToPool = { 0 };
    pcfg.classNames = m_impl->final_cfg.classNames;
    pcfg.resultQueueSize =
        (m_impl->params.result_queue_size > 0)
            ? static_cast<std::size_t>(m_impl->params.result_queue_size)
            : std::size_t{32};

    if (render) {
        pcfg.renderer = &m_impl->renderer;
        pcfg.saveEnabled = m_impl->params.save;
        pcfg.showEnabled = m_impl->params.show;
        // 存盘目录：params.save_dir > ini output.save_dir > 默认 save/<net>。
        // 谁显式给了就用谁的（原样，不拼子目录）。
        std::string saveDir = m_impl->params.save_dir;
        if (saveDir.empty()) {
            saveDir = m_impl->pool->modelConfig().getString("output.save_dir", "");
        }
        pcfg.saveDir = core::Paths::resolveSaveDir(
            saveDir, registry_name(m_impl->params.model_type));
        if (!m_impl->params.show_window.empty()) pcfg.showWindow = m_impl->params.show_window;
    } else {
        pcfg.renderer = nullptr;
        pcfg.saveEnabled = false;
        pcfg.showEnabled = false;
    }

    Stream s;
    s.m_impl = std::make_unique<Stream::Impl>();
    // 先保活再建 Pipeline：Pipeline 会记下 pool / renderer 的裸指针，
    // 之后即使 Infer 析构，只要 Stream 还在，这些资源就不会被释放。
    s.m_impl->owner = m_impl;
    s.m_impl->pipeline = std::make_unique<pipeline::Pipeline>(std::move(pcfg));
    s.m_impl->pipeline->start();
    return s;
}

}  // namespace trt_alpha