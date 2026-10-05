// =============================================================================
//  trt_alpha :: infer（实现）
// =============================================================================
#include "trt_alpha/infer/infer.hpp"

#include "trt_alpha/core/config.hpp"
#include "trt_alpha/core/inference_pool.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/datasource/opencv_source.hpp"
#include "trt_alpha/pipeline/pipeline.hpp"
#include "trt_alpha/pipeline/pipeline_config.hpp"
#include "trt_alpha/renderer/opencv_renderer.hpp"

#include <cstdio>
#include <stdexcept>
#include <utility>

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

//! 构造数据源（根据 source / camera_id 推断类型）。
datasource::SourceConfig build_source_config(const InferParams& p, int batch_size)
{
    datasource::SourceConfig cfg;
    cfg.batchSize = batch_size;
    cfg.sourceId = 0;
    cfg.loop = p.loop;

    if (!p.source.empty())
    {
        // 判断源类型：视频 / URL / 图片
        const std::string& s = p.source;
        const bool is_video =
            s.find(".mp4") != std::string::npos ||
            s.find(".avi") != std::string::npos ||
            s.find(".mov") != std::string::npos ||
            s.find("rtsp://") == 0 ||
            s.find("http://") == 0  ||
            s.find("https://") == 0;

        if (is_video) {
            cfg.type = datasource::SourceType::Video;
            cfg.path = s;
        } else {
            cfg.type = datasource::SourceType::Image;
            cfg.path = s;
        }
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

    void print_config() const;
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

void Infer::Impl::print_config() const
{
    TRT_LOG_INFO("Infer: final config");
    TRT_LOG_INFO("  model_type         = " << registry_name(params.model_type));
    TRT_LOG_INFO("  config_path        = " << params.config_path);
    TRT_LOG_INFO("  engine             = " << final_cfg.engine
                 << (params.engine.empty() ? "" : "  (override)"));
    TRT_LOG_INFO("  class_names_file   = " << final_cfg.classNamesFile
                 << (params.class_names_file.empty() ? "" : "  (override)"));
    TRT_LOG_INFO("  batch_size         = " << final_cfg.batchSize
                 << (params.batch_size > 0 ? "  (override)" : ""));
    TRT_LOG_INFO("  dst_h / dst_w      = " << final_cfg.dstH << " / " << final_cfg.dstW
                 << ((params.dst_h > 0 || params.dst_w > 0) ? "  (override)" : ""));
    TRT_LOG_INFO("  conf_thresh        = " << final_cfg.getFloat("conf_thresh", 0.f)
                 << (params.conf_thresh >= 0.f ? "  (override)" : ""));
    TRT_LOG_INFO("  iou_thresh         = " << final_cfg.getFloat("iou_thresh", 0.f)
                 << (params.iou_thresh >= 0.f ? "  (override)" : ""));
    TRT_LOG_INFO("  top_k              = " << final_cfg.getInt("top_k", 0)
                 << (params.top_k > 0 ? "  (override)" : ""));
    TRT_LOG_INFO("  source             = "
                 << (params.source.empty() ? "(camera)" : params.source));
    TRT_LOG_INFO("  camera_id          = " << params.camera_id);
    TRT_LOG_INFO("  save / show        = " << params.save << " / " << params.show);
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
}

// =============================================================================
//  Infer
// =============================================================================
Infer::Infer(const InferParams& p)
    : m_impl(std::make_unique<Impl>())
{
    p.validate();
    m_impl->params = p;
    m_impl->final_cfg = m_impl->merge_config();
    m_impl->print_config();
    // 懒建 pool（首次 run/async 时建）
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
        if (!m_impl->params.save_dir.empty())    pcfg.saveDir = m_impl->params.save_dir;
        if (!m_impl->params.show_window.empty()) pcfg.showWindow = m_impl->params.show_window;
    } else {
        pcfg.renderer = nullptr;
        pcfg.saveEnabled = false;
        pcfg.showEnabled = false;
    }

    Stream s;
    s.m_impl = std::make_unique<Stream::Impl>();
    s.m_impl->pipeline = std::make_unique<pipeline::Pipeline>(std::move(pcfg));
    s.m_impl->pipeline->start();
    return s;
}

}  // namespace trt_alpha