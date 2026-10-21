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
#include "trt_alpha/datasource/source_config.hpp"
#include "trt_alpha/datasource/source_factory.hpp"
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
    "yolov3", "yolov4", "yolov5", "yolov6", "yolov7", "yolov8_old",
    "yolov8_pose_old", "yolov8_seg_old",
    "yolox", "yolor", "yolo_nas",
    "efficientdet", "yunet", "u2net",
    "yolov26",
    // YOLOV26 task variants -- order must match ModelType (append-only).
    "yolov26_cls", "yolov26_depth", "yolov26_obb",
    "yolov26_pose", "yolov26_seg", "yolov26_sem",
    // YOLOV8 on the fused pipeline -- order must match ModelType.
    "yolov8", "yolov8_pose", "yolov8_seg", "yolov8_cls", "yolov8_obb",
    // YOLOV9 / YOLOV10 on the fused pipeline -- order must match ModelType.
    "yolov9", "yolov9_seg", "yolov10",
    // YOLOV11 on the fused pipeline -- order must match ModelType.
    "yolov11", "yolov11_cls", "yolov11_obb", "yolov11_pose", "yolov11_seg",
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

//! QueuePolicy（对外）→ QueueFullPolicy（队列实现）。
core::QueueFullPolicy to_queue_policy(QueuePolicy p, bool realtime)
{
    switch (p) {
    case QueuePolicy::DropOldest: return core::QueueFullPolicy::DropOldest;
    case QueuePolicy::DropNewest: return core::QueueFullPolicy::DropNewest;
    case QueuePolicy::Block:      return core::QueueFullPolicy::Block;
    case QueuePolicy::Unset:
    default:
        // 未指定 → 按源类型自动：实时源宁可丢旧帧，离线源一帧不落。
        return realtime ? core::QueueFullPolicy::DropOldest
                        : core::QueueFullPolicy::Block;
    }
}

//! 构造数据源（根据 source / camera_id 推断类型）。
//! source 的判别交给 datasource::resolveSource —— 与 CLI `--source` 共用同一套
//! 规则（URL / 视频后缀 / 纯数字摄像头 / 其余=图片），避免两处各写一份慢慢走偏。
//!
//! 解码相关的键从 ini 的 `[source]` 段读，优先级 InferParams > ini > 代码默认
//! （与 CLI 的 `--decode` > ini > 默认 是同一条链）。
//! batch_size 只是占位：async() 会换成池解析后的真实 batch —— 静态引擎会忽略
//! 请求值、动态引擎越界直接报错，按原值攒批会写越界。
datasource::SourceConfig build_source_config(const InferParams& p,
                                             const core::ModelConfig& ini,
                                             int batch_size)
{
    datasource::SourceConfig cfg;
    cfg.batchSize = batch_size;
    cfg.sourceId = 0;
    cfg.loop = p.loop;

    const std::string decodeName =
        p.decode.empty() ? ini.getString("source.decode", "opencv") : p.decode;
    const std::string hwaccelName =
        p.hwaccel.empty() ? ini.getString("source.hwaccel", "auto") : p.hwaccel;

    cfg.decodeBackend = datasource::requireDecodeBackend(decodeName);
    cfg.hwAccel = datasource::requireHwAccel(hwaccelName);

    // 0 是合法值（= 交给后端默认），所以这里按 >= 0 判"用户给了"，-1 才是用 ini。
    cfg.openTimeoutMs = (p.open_timeout_ms >= 0)
                            ? p.open_timeout_ms
                            : ini.getInt("source.open_timeout_ms", 5000);
    cfg.readTimeoutMs = (p.read_timeout_ms >= 0)
                            ? p.read_timeout_ms
                            : ini.getInt("source.read_timeout_ms", 5000);

    if (!p.source.empty())
    {
        const datasource::ResolvedSource rs = datasource::resolveSource(p.source);
        cfg.type     = rs.type;
        cfg.path     = rs.path;
        cfg.cameraId = rs.cameraId;
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
    if (!validSlot(m_result == nullptr ? 0 : m_result->views.size())) {
        return {};
    }
    const core::BufferView& v = m_result->views[m_index];
    // With draw / show / save all off, a source may skip its host copy and then
    // these views describe DEVICE memory (see SourceConfig::needHostCopy). Handing
    // that back as an image would be a pointer the caller must not wrap in cv::Mat
    // or feed to imshow, so report "no image" instead of a trap. The detections /
    // masks / depth are unaffected -- only the pixels are gone.
    if (v.space != core::MemorySpace::Host) {
        return {};
    }
    return v;
}

core::BufferView Frame::rendered() const
{
    if (m_result == nullptr || m_result->renderedViews.empty()) {
        return {};
    }
    if (!validSlot(m_result->renderedViews.size())) {
        return {};
    }
    const core::BufferView& v = m_result->renderedViews[m_index];
    // Same guard: a rendered copy only exists when a renderer ran, so this is
    // host memory in practice -- but never hand out a device pointer as an image.
    if (v.space != core::MemorySpace::Host) {
        return {};
    }
    return v;
}

const std::vector<det::Detection>& Frame::boxes() const
{
    static const std::vector<det::Detection> kEmpty;
    if (!validSlot(m_result == nullptr ? 0 : m_result->detections.size())) {
        return kEmpty;
    }
    return m_result->detections[m_index];
}

const std::vector<seg::Segmentation>& Frame::masks() const
{
    static const std::vector<seg::Segmentation> kEmpty;
    if (!validSlot(m_result == nullptr ? 0 : m_result->segmentations.size())) {
        return kEmpty;
    }
    return m_result->segmentations[m_index];
}

const std::vector<kpt::KeypointResult>& Frame::keypoints() const
{
    static const std::vector<kpt::KeypointResult> kEmpty;
    if (!validSlot(m_result == nullptr ? 0 : m_result->keypoints.size())) {
        return kEmpty;
    }
    return m_result->keypoints[m_index];
}

const std::vector<cls::ClassScore>& Frame::classifications() const
{
    static const std::vector<cls::ClassScore> kEmpty;
    if (!validSlot(m_result == nullptr ? 0 : m_result->classifications.size())) {
        return kEmpty;
    }
    return m_result->classifications[m_index];
}

const std::vector<obb::ObbDetection>& Frame::obbs() const
{
    static const std::vector<obb::ObbDetection> kEmpty;
    if (!validSlot(m_result == nullptr ? 0 : m_result->obbs.size())) {
        return kEmpty;
    }
    return m_result->obbs[m_index];
}

const depth::DepthMap& Frame::depth() const
{
    static const depth::DepthMap kEmpty;
    if (!validSlot(m_result == nullptr ? 0 : m_result->depths.size())) {
        return kEmpty;
    }
    return m_result->depths[m_index];
}

const sem::SemanticMask& Frame::semantic() const
{
    static const sem::SemanticMask kEmpty;
    if (!validSlot(m_result == nullptr ? 0 : m_result->semantics.size())) {
        return kEmpty;
    }
    return m_result->semantics[m_index];
}

// =============================================================================
//  Result
// =============================================================================
std::vector<Frame> Result::frames() const
{
    const std::size_t n = size();          // valid frames only (skip zero-fill padding)
    std::vector<Frame> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.emplace_back(&m_result, i);
    }
    return out;
}

std::vector<det::Detection> Result::boxes() const
{
    std::vector<det::Detection> out;
    const std::size_t n = size();
    for (std::size_t i = 0; i < n && i < m_result.detections.size(); ++i) {
        const auto& v = m_result.detections[i];
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<seg::Segmentation> Result::masks() const
{
    std::vector<seg::Segmentation> out;
    const std::size_t n = size();
    for (std::size_t i = 0; i < n && i < m_result.segmentations.size(); ++i) {
        const auto& v = m_result.segmentations[i];
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<kpt::KeypointResult> Result::keypoints() const
{
    std::vector<kpt::KeypointResult> out;
    const std::size_t n = size();
    for (std::size_t i = 0; i < n && i < m_result.keypoints.size(); ++i) {
        const auto& v = m_result.keypoints[i];
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<cls::ClassScore> Result::classifications() const
{
    std::vector<cls::ClassScore> out;
    const std::size_t n = size();
    for (std::size_t i = 0; i < n && i < m_result.classifications.size(); ++i) {
        const auto& v = m_result.classifications[i];
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<obb::ObbDetection> Result::obbs() const
{
    std::vector<obb::ObbDetection> out;
    const std::size_t n = size();
    for (std::size_t i = 0; i < n && i < m_result.obbs.size(); ++i) {
        const auto& v = m_result.obbs[i];
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<depth::DepthMap> Result::depths() const
{
    const std::size_t n = std::min(size(), m_result.depths.size());
    return std::vector<depth::DepthMap>(m_result.depths.begin(),
                                        m_result.depths.begin() + n);
}

std::vector<sem::SemanticMask> Result::semantics() const
{
    const std::size_t n = std::min(size(), m_result.semantics.size());
    return std::vector<sem::SemanticMask>(m_result.semantics.begin(),
                                          m_result.semantics.begin() + n);
}

// =============================================================================
//  Infer::Impl
// =============================================================================
struct Infer::Impl
{
    InferParams params;
    core::ModelConfig final_cfg;
    //! 源的解码选项（不含 batch / 回拷决策）：构造期就从 ini 读好，这样
    //! readKeys 会随 final_cfg 一起拷进池，配置框不会把 [source] 的键误标
    //! 成 [unused]。async() 只补 batchSize 和 needHostCopy。
    datasource::SourceConfig src_template;
    //! need_host_copy 的三态（auto 要等池建好、知道模型能力才能落地）。
    datasource::HostCopy host_mode = datasource::HostCopy::Auto;
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

    if (!params.save.dir.empty())    cfg.extras["save_dir"] = params.save.dir;
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

    // 能走到这里说明 batch 已通过引擎 profile 校验（不符在池构造期就抛了），
    // 所以它必然等于请求值 —— 不再需要"被纠正"提示。
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
    // 但 [source] 的键必须现在就读完：池构造时才拷贝 final_cfg，晚读的键
    // 不会进池副本，配置框会把它们标成 [unused]（看着像没生效）。
    m_impl->src_template = build_source_config(p, m_impl->final_cfg, 1);
    const std::string hostCopyName =
        p.need_host_copy.empty()
            ? m_impl->final_cfg.getString("source.need_host_copy", "auto")
            : p.need_host_copy;
    m_impl->host_mode = datasource::requireHostCopy(hostCopyName);
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

    // Draw when the caller asked for the annotated frame (draw), or when the result
    // needs to be written / shown -- all three share the same renderer + copy path.
    const bool render = m_impl->params.draw ||
                        m_impl->params.show ||
                        m_impl->params.save.enabled;

    datasource::SourceConfig srcCfg = m_impl->src_template;
    // batch 取池解析后的真实值（模板里那个 1 只是占位）。
    srcCfg.batchSize = m_impl->pool->resolvedBatch();

    const bool devicePixels = m_impl->pool->acceptsDevicePixels();

    // Same rule as the CLI run command (see SourceConfig::needHostCopy): the host
    // copy exists so the renderer can draw into it. With draw / show / save all
    // off nothing reads host pixels, and a hardware decoder can leave them in
    // device memory -- 0.475 ms of a 1.70 ms 1080p frame. A model that still
    // uploads from the host needs the copy regardless, hence the pool check.
    // need_host_copy turns the two explicit overrides on top of that.
    srcCfg.needHostCopy = render || !devicePixels;
    switch (m_impl->host_mode)
    {
    case datasource::HostCopy::On:
        // 显式要求：即使没人读也留着（调用方要拿 Frame::image()）。
        srcCfg.needHostCopy = true;
        break;
    case datasource::HostCopy::Off:
        // 这两条中任一条成立时"不拷"根本不可能满足：渲染器会把显存指针当主机
        // 图像画，模型会在空缓冲上做 H2D。报错，绝不悄悄改回 true。
        if (render)
        {
            throw std::runtime_error(
                "InferParams: need_host_copy = off cannot be honoured together with "
                "draw / show / save (the renderer draws into the host copy); "
                "use auto or on");
        }
        if (!devicePixels)
        {
            throw std::runtime_error(
                "InferParams: need_host_copy = off cannot be honoured with this model "
                "(it uploads pixels from the host); use auto or on");
        }
        srcCfg.needHostCopy = false;
        break;
    case datasource::HostCopy::Auto:
    default:
        break;
    }

    TRT_LOG_INFO("Infer: host pixel copy " << (srcCfg.needHostCopy ? "ON" : "OFF")
                 << " (render=" << (render ? "on" : "off")
                 << ", model " << (devicePixels ? "supports" : "does not support")
                 << " device pixels, decode="
                 << datasource::toString(srcCfg.decodeBackend)
                 << ", hwaccel=" << datasource::toString(srcCfg.hwAccel)
                 << ", need_host_copy=" << datasource::toString(m_impl->host_mode)
                 << ")");
    const bool realtime =
        (srcCfg.type == datasource::SourceType::Camera) ||
        core::Paths::isUrl(m_impl->params.source);

    pipeline::PipelineConfig pcfg;
    pcfg.sources.push_back(datasource::makeSource(srcCfg));
    pcfg.pools = { m_impl->pool.get() };
    pcfg.sourceToPool = { 0 };
    pcfg.classNames = m_impl->final_cfg.classNames;
    pcfg.resultQueueSize =
        (m_impl->params.result_queue_size > 0)
            ? static_cast<std::size_t>(m_impl->params.result_queue_size)
            : std::size_t{32};
    // 队列满策略：显式指定优先，否则按源类型自动（实时丢旧帧 / 离线不丢帧）
    pcfg.queueFullPolicy = to_queue_policy(m_impl->params.policy, realtime);

    if (render) {
        pcfg.renderer = &m_impl->renderer;
        // Stream::get() 在有渲染器时走 popProcessed ⇒ 必须先声明要消费
        pcfg.exposeProcessed = true;
        pcfg.saveEnabled = m_impl->params.save.enabled;
        pcfg.showEnabled = m_impl->params.show;
        // 存盘目录：params.save.dir > ini output.save_dir > 默认 save/<net>。
        // 谁显式给了就用谁的（原样，不拼子目录）。
        std::string saveDir = m_impl->params.save.dir;
        if (saveDir.empty()) {
            saveDir = m_impl->pool->modelConfig().getString("output.save_dir", "");
        }
        pcfg.saveDir = core::Paths::resolveSaveDir(
            saveDir, registry_name(m_impl->params.model_type));
        // 窗口名：params.show_window > ini output.show_window > "trt_alpha:<net>"。
        // 默认带注册名，和 run 命令保持同一口径（多模型同开时窗口不互相覆盖）。
        const std::string iniWindow =
            m_impl->pool->modelConfig().getString("output.show_window", "");
        if (!m_impl->params.show_window.empty()) {
            pcfg.showWindow = m_impl->params.show_window;
        } else if (!iniWindow.empty()) {
            pcfg.showWindow = iniWindow;
        } else {
            pcfg.showWindow = std::string("trt_alpha:") + registry_name(m_impl->params.model_type);
        }

        // 登记输入源：save() 据此拒绝覆盖原图（输出 == 输入的最后一道闸门）
        if (m_impl->params.save.enabled && !core::Paths::isUrl(m_impl->params.source)) {
            m_impl->renderer.addInputSource(
                core::Paths::resolve(m_impl->params.source).string());
        }
    } else {
        pcfg.renderer = nullptr;
        pcfg.exposeProcessed = false;
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
