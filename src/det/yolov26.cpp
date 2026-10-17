// =============================================================================
//  trt_alpha :: det :: YoloV26（实现）
// -----------------------------------------------------------------------------
//  管线：
//      setBatch   : 原图连续内存 H2D
//      preprocess : CvPreprocessVecPlan  —— letterbox 640x640 + BGR2RGB +
//                   HWC2CHW + /255，五合一单 kernel
//      infer      : enqueueV3
//      postprocess: CvProprocessFusedVecYoloV26Plan —— decode + conf 过滤 +
//                   NMS，单 kernel；然后 scale_boxes 还原到原图坐标
//
//  为什么要用 kernels/ops 而不是 legacy 的 decodeYoloV8Head + nmsFast：
//      YOLOV26 的 ONNX 已经把解码烘焙进图里（output0 = 已解码的 y），legacy
//      那套会再解一次，而且 IoU 口径不同（分母没有 +1e-9）。ops 这套与
//      ultralytics 参考逐位一致（max|d| = 0，已在 9 张 VOC val 图上验证）。
// =============================================================================
#include "yolov26.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"   // TRT_ALPHA_REGISTER_MODEL

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trt_alpha::det {
namespace {

//! ultralytics NMS 的类别偏移基数（固定值，与 python 参考一致）。
constexpr float kMaxWh = 7680.0f;

}  // namespace

const std::string& YoloV26::name() const noexcept
{
    static const std::string kName = "yolov26";
    return kName;
}

void YoloV26::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    if (m_numClass <= 0)
    {
        throw std::runtime_error("yolov26: num_class must be > 0 (got " +
                                 std::to_string(m_numClass) + ")");
    }

    // extra: 每 anchor 的额外通道（seg 32 / pose 51）。本次只接 det，默认 0；
    // 留成 INI 可配项，将来接 seg/pose 不用改模型。
    m_extra = cfg.getInt("extra", 0);
    if (m_extra < 0)
    {
        throw std::runtime_error("yolov26: extra must be >= 0 (got " +
                                 std::to_string(m_extra) + ")");
    }

    if (m_topK <= 0)
    {
        throw std::runtime_error("yolov26: top_k must be > 0 (got " +
                                 std::to_string(m_topK) + ")");
    }

    TRT_LOG_INFO("YoloV26: config num_class=" << m_numClass
                 << " extra=" << m_extra
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK);
}

void YoloV26::discoverEngineIo()
{
    // 按名字解析（不依赖 binding 顺序）
    const core::TensorDesc* input = nullptr;
    const core::TensorDesc* output = nullptr;
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput && input == nullptr)   { input = &t; }
        if (!t.isInput && output == nullptr) { output = &t; }
    }
    if (input == nullptr || output == nullptr)
    {
        throw std::runtime_error("yolov26: engine must have >=1 input and >=1 output");
    }
    m_inputName = input->name;
    m_outputName = output->name;

    // 解析并下发输入形状：以引擎声明形状为真相源（H/W 静态维自动纠正）
    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 3)
    {
        throw std::runtime_error("yolov26: expect output0 as [batch, 4+nc+extra, anchors]");
    }
    if (outDims.d[0] > 0 && outDims.d[0] != m_cfg.batchSize)
    {
        throw std::runtime_error(
            "yolov26: output0 batch = " + std::to_string(outDims.d[0]) +
            " but input batch = " + std::to_string(m_cfg.batchSize) +
            " (engine profile is the single source of truth; check 'batch_size')");
    }

    m_outC    = static_cast<int>(outDims.d[1]);
    m_anchors = static_cast<int>(outDims.d[2]);

    const int expectC = 4 + m_numClass + m_extra;
    if (m_outC != expectC)
    {
        throw std::runtime_error(
            "yolov26: output0 channel = " + std::to_string(m_outC) +
            " but 4+num_class+extra = " + std::to_string(expectC) +
            " (check 'num_class' / 'extra' in INI)");
    }
    if (m_anchors <= 0)
    {
        throw std::runtime_error("yolov26: output0 anchors must be > 0 (got " +
                                 std::to_string(m_anchors) + ")");
    }
}

void YoloV26::buildPlans()
{
    // 后处理的 anchors / 类别数只有在引擎形状落定之后才知道，所以 Plan 在这里建。
    kernels::ops::ProprocessFusedVecYoloV26Config pcfg;
    pcfg.anchors    = m_anchors;
    pcfg.nc         = m_numClass;
    pcfg.extra      = m_extra;
    pcfg.maxBatch   = m_cfg.batchSize;
    pcfg.maxDet     = m_topK;
    pcfg.confThresh = m_confThreshold;
    pcfg.iouThresh  = m_iouThreshold;
    pcfg.maxWh      = kMaxWh;

    m_pp = std::make_unique<kernels::ops::CvProprocessFusedVecYoloV26Plan>(pcfg);
    if (!m_pp->ok())
    {
        throw std::runtime_error("yolov26: fused postprocess plan construction failed");
    }
}

void YoloV26::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;

    // --- Device buffer 分配，每个打框图 log ---
    auto logBox = [&](const char* nm, int batch, int ch, int h, int w,
                      core::DataType dt, std::size_t bytes, core::MemorySpace sp)
    {
        core::detail::AllocInfo info;
        info.name     = nm;
        info.batch    = batch;
        info.channels = ch;
        info.height   = h;
        info.width    = w;
        info.dtype    = dt;
        info.bytes    = bytes;
        info.space    = sp;
        core::detail::logAllocBox(info);
    };

    m_inputSrc.allocate(static_cast<std::size_t>(B) * 3 * W * H);
    logBox("yolov26.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * 3 * W * H * sizeof(float));
    logBox("yolov26.input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(static_cast<std::size_t>(B) * m_outC * m_anchors * sizeof(float));
    logBox("yolov26.output_src", B, m_outC, 1, m_anchors, core::DataType::Float32,
           m_outputSrc.bytes(), core::MemorySpace::Device);

    const int detStride = m_pp->detStride();
    const std::size_t detBytes =
        static_cast<std::size_t>(B) * m_topK * detStride * sizeof(float);
    const std::size_t cntBytes = static_cast<std::size_t>(B) * sizeof(int);

    m_dets.allocate(detBytes);
    logBox("yolov26.dets", B, m_topK, 1, detStride, core::DataType::Float32,
           m_dets.bytes(), core::MemorySpace::Device);

    m_counts.allocate(cntBytes);
    logBox("yolov26.counts", B, 1, 1, 1, core::DataType::Int32,
           m_counts.bytes(), core::MemorySpace::Device);

    m_detsHost.allocate(detBytes);
    logBox("yolov26.dets_host", B, m_topK, 1, detStride, core::DataType::Float32,
           m_detsHost.bytes(), core::MemorySpace::Host);

    m_countsHost.allocate(cntBytes);
    logBox("yolov26.counts_host", B, 1, 1, 1, core::DataType::Int32,
           m_countsHost.bytes(), core::MemorySpace::Host);

    // ---- 绑定 I/O 张量地址（buffer 指针不变，一次即可）----
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()))
    {
        throw std::runtime_error("yolov26: setTensorAddress failed");
    }
}

void YoloV26::init(const core::ModelConfig& cfg)
{
    TRT_LOG_DEBUG("YoloV26::init: entering");
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    TRT_LOG_DEBUG("YoloV26::init: engine loaded");
    discoverEngineIo();
    TRT_LOG_DEBUG("YoloV26::init: io discovered (outC=" << m_outC
                 << ", anchors=" << m_anchors << ")");
    buildPlans();
    TRT_LOG_DEBUG("YoloV26::init: plans built (detStride=" << m_pp->detStride() << ")");
    allocateBuffers();
    TRT_LOG_DEBUG("YoloV26::init: buffers allocated");
    m_detections.assign(static_cast<std::size_t>(m_cfg.batchSize), {});

    TRT_LOG_INFO("YoloV26: initialized (batch=" << m_cfg.batchSize
                 << ", outC=" << m_outC
                 << ", anchors=" << m_anchors
                 << ", dst=" << m_cfg.dstW << "x" << m_cfg.dstH << ")");
}

void YoloV26::ensurePreprocessPlan()
{
    // 几何没变就复用（同一个视频/同尺寸图目录下不会反复重建）
    if (m_pre && m_pre->matches(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH))
    {
        m_lb = m_pre->info();
        return;
    }

    m_pre = std::make_unique<kernels::ops::CvPreprocessVecPlan>(
        kernels::ops::CvPreprocessVecPlan::byDivisor(
            m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH,
            static_cast<double>(m_padValue), m_normScale));
    if (!m_pre->ok())
    {
        throw std::runtime_error("yolov26: preprocess plan construction failed");
    }
    m_lb = m_pre->info();

    TRT_LOG_DEBUG("YoloV26: letterbox rebuilt  " << m_srcW << "x" << m_srcH
                 << " -> new " << m_lb.newW << "x" << m_lb.newH
                 << " pad(l/r/t/b)=" << m_lb.padLeft << "/" << m_lb.padRight
                 << "/" << m_lb.padTop << "/" << m_lb.padBottom
                 << " ratio=" << m_lb.ratio);
}

void YoloV26::setBatch(const core::Batch& batch)
{
    TRT_LOG_DEBUG("YoloV26::setBatch: entering, views=" << batch.views.size()
                 << ", validCount=" << batch.validCount);
    if (batch.views.empty())
    {
        throw std::runtime_error("yolov26: empty batch");
    }

    m_batch = requireBatchCapacity(*this, batch, "yolov26");
    m_srcH  = batch.views[0].height;
    m_srcW  = batch.views[0].width;
    if (m_srcW <= 0 || m_srcH <= 0)
    {
        throw std::runtime_error("yolov26: invalid source size " +
                                 std::to_string(m_srcW) + "x" + std::to_string(m_srcH));
    }

    // 正/反变换同源：几何一变就重建 Plan，同时刷新 m_lb
    ensurePreprocessPlan();

    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolov26: batch.buffer is null");
    }

    const std::size_t oneImage = 3 * static_cast<std::size_t>(m_srcH) * m_srcW;
    const std::size_t total    = oneImage * batch.views.size();

    if (total > m_inputSrc.bytes())
    {
        // 源图尺寸变化触发扩容（少见）
        m_inputSrc.allocate(total);
    }

    cudaMemcpyAsync(m_inputSrc.data(), batch.buffer->data(), total,
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();

    TRT_LOG_DEBUG("YoloV26::setBatch: H2D done, " << total << " bytes (batch=" << m_batch << ")");
}

void YoloV26::preprocess()
{
    if (!m_pre)
    {
        throw std::runtime_error("yolov26: preprocess called before setBatch");
    }
    m_pre->launch(m_stream.get(), m_batch,
                  static_cast<const unsigned char*>(m_inputSrc.data()),
                  m_inputNchw.asFloat());
    if (!m_pre->ok())
    {
        throw std::runtime_error("yolov26: preprocess kernel launch failed");
    }
#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_DEBUG
    // 分阶段计时专用：不在这里等，preprocess 只能量到"入队"，letterbox 的 GPU
    // 时间会一路拖到 postprocess 末尾那次同步，三段就没法拆开。仅 DEBUG 构建生效，
    // Release 下是零成本（不影响语义，后面那次同步仍然照做）。
    m_stream.synchronize();
#endif
    TRT_LOG_DEBUG("YoloV26::preprocess: done (batch=" << m_batch << ")");
}

void YoloV26::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    // 地址已在 allocateBuffers() 绑定
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolov26: enqueueV3 failed");
    }
#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_DEBUG
    // 同上：等引擎真正跑完，infer 这一段才是网络执行时间（含 H2D 后的排队）。
    m_stream.synchronize();
#endif
    TRT_LOG_DEBUG("YoloV26::infer: enqueueV3 done (batch=" << m_batch << ")");
}

void YoloV26::postprocess()
{
    // commitResult 会把 m_detections move 走，所以每轮重新铺尺寸
    m_detections.assign(static_cast<std::size_t>(m_batch), {});

    m_pp->launch(m_stream.get(), m_batch, m_outputSrc.asFloat(),
                 static_cast<int*>(m_counts.data()), m_dets.asFloat());
    if (!m_pp->ok())
    {
        throw std::runtime_error("yolov26: fused postprocess kernel launch failed");
    }

    const int         stride   = m_pp->detStride();
    const std::size_t cntBytes = static_cast<std::size_t>(m_batch) * sizeof(int);
    const std::size_t detBytes = static_cast<std::size_t>(m_batch) * m_topK * stride * sizeof(float);

    cudaMemcpyAsync(m_countsHost.data(), m_counts.data(), cntBytes,
                    cudaMemcpyDeviceToHost, m_stream.get());
    cudaMemcpyAsync(m_detsHost.data(), m_dets.data(), detBytes,
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    const int*   counts = static_cast<const int*>(m_countsHost.data());
    const float* dets   = m_detsHost.asFloat();

    const double srcW = static_cast<double>(m_srcW);
    const double srcH = static_cast<double>(m_srcH);

    for (int b = 0; b < m_batch; ++b)
    {
        const int n = std::clamp(counts[b], 0, m_topK);
        const float* row = dets + static_cast<std::size_t>(b) * m_topK * stride;
        auto& out = m_detections[static_cast<std::size_t>(b)];
        out.clear();
        out.reserve(static_cast<std::size_t>(n));

        for (int i = 0; i < n; ++i)
        {
            const float* d = row + static_cast<std::size_t>(i) * stride;
            Detection det;
            // 网络输入坐标 -> 原图坐标，并裁进图像范围（同 ultralytics clip_boxes）
            det.left   = static_cast<float>(
                std::min(std::max(m_lb.toOrigX(d[0]), 0.0), srcW));
            det.top    = static_cast<float>(
                std::min(std::max(m_lb.toOrigY(d[1]), 0.0), srcH));
            det.right  = static_cast<float>(
                std::min(std::max(m_lb.toOrigX(d[2]), 0.0), srcW));
            det.bottom = static_cast<float>(
                std::min(std::max(m_lb.toOrigY(d[3]), 0.0), srcH));
            det.confidence = d[4];
            det.label      = static_cast<int>(d[5]);
            out.push_back(det);
        }
    }

    for (int b = 0; b < m_batch; ++b)
    {
        TRT_LOG_DEBUG("YoloV26::postprocess: batch[" << b << "] detections="
                     << m_detections[static_cast<std::size_t>(b)].size());
    }
}

void YoloV26::reset()
{
    TRT_LOG_DEBUG("YoloV26::reset: clearing " << m_detections.size() << " image result(s)");
    for (auto& v : m_detections)
    {
        v.clear();
    }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("yolov26", trt_alpha::det::YoloV26);
