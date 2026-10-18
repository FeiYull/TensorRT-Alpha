// =============================================================================
//  trt_alpha :: obb :: YoloV26Obb（实现）
// -----------------------------------------------------------------------------
//  管线：
//      setBatch   : 原图连续内存 H2D
//      preprocess : CvPreprocessVecPlan —— letterbox 640x640 + BGR2RGB +
//                   HWC2CHW + /255（与 det 逐位相同的那个融合核）
//      infer      : enqueueV3
//      postprocess: CvPostprocessFusedYoloV26ObbPlan —— decode 已在图里，算子做
//                   conf 过滤 + 稳定排序 + **Fast-NMS(probiou)**；随后 CPU 侧按
//                   ops.scale_boxes(..., xywh=True) 把中心与边长还原回原图
//
//  参考实现：ultralytics/models/yolo/obb/predict.py::construct_result
//            ultralytics/utils/nms.py::non_max_suppression（rotated=True 分支）
// =============================================================================
#include "yolov26_obb.hpp"

#include "trt_alpha/core/config.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"   // TRT_ALPHA_REGISTER_MODEL

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace trt_alpha::obb {
namespace {

//! ultralytics NMS 的类别偏移基数（固定值，与 python 参考一致）。
constexpr float kMaxWh = 7680.0f;

//! The rotated branch always reads the angle from the extra block; the yolo26
//! OBB head carries exactly one extra channel.
constexpr int kObbExtra = 1;

}  // namespace

const std::string& YoloV26Obb::name() const noexcept
{
    return m_tag;
}

void YoloV26Obb::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    // The fused preprocess applies 'scale' only; mean/std must not pass silently.
    core::requireScaleOnlyNorm(cfg, m_tag);

    if (m_numClass <= 0)
    {
        throw std::runtime_error(m_tag + ": num_class must be > 0 (got " +
                                 std::to_string(m_numClass) + ")");
    }
    if (m_topK <= 0)
    {
        throw std::runtime_error(m_tag + ": top_k must be > 0 (got " +
                                 std::to_string(m_topK) + ")");
    }

    const int extra = cfg.getInt("extra", kObbExtra);
    if (extra != kObbExtra)
    {
        throw std::runtime_error(
            m_tag + ": extra must be " + std::to_string(kObbExtra) +
            " (the rotation angle), got " + std::to_string(extra));
    }

    TRT_LOG_INFO(m_tag << ": config num_class=" << m_numClass
                 << " extra=" << extra
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK);
}

void YoloV26Obb::discoverEngineIo()
{
    const core::TensorDesc* input  = nullptr;
    const core::TensorDesc* output = nullptr;
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput && input == nullptr)   { input = &t; }
        if (!t.isInput && output == nullptr) { output = &t; }
    }
    if (input == nullptr || output == nullptr)
    {
        throw std::runtime_error(m_tag + ": engine must have >=1 input and >=1 output");
    }
    m_inputName  = input->name;
    m_outputName = output->name;

    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 3)
    {
        throw std::runtime_error(m_tag + ": expect output0 as [batch, 4+nc+1, anchors]");
    }
    if (outDims.d[0] > 0 && outDims.d[0] != m_cfg.batchSize)
    {
        throw std::runtime_error(
            m_tag + ": output0 batch = " + std::to_string(outDims.d[0]) +
            " but input batch = " + std::to_string(m_cfg.batchSize) +
            " (engine profile is the single source of truth; check 'batch_size')");
    }

    m_outC    = static_cast<int>(outDims.d[1]);
    m_anchors = static_cast<int>(outDims.d[2]);

    const int expectC = 4 + m_numClass + kObbExtra;
    if (m_outC != expectC)
    {
        throw std::runtime_error(
            m_tag + ": output0 channel = " + std::to_string(m_outC) +
            " but 4+num_class+1 = " + std::to_string(expectC) +
            " (check 'num_class' in INI)");
    }
    if (m_anchors <= 0)
    {
        throw std::runtime_error(m_tag + ": output0 anchors must be > 0 (got " +
                                 std::to_string(m_anchors) + ")");
    }
}

void YoloV26Obb::buildPlans()
{
    kernels::ops::ProprocessFusedYoloV26Config pcfg;
    pcfg.anchors    = m_anchors;
    pcfg.nc         = m_numClass;
    pcfg.extra      = kObbExtra;
    pcfg.maxBatch   = m_cfg.batchSize;
    pcfg.maxDet     = m_topK;
    pcfg.confThresh = m_confThreshold;
    pcfg.iouThresh  = m_iouThreshold;
    pcfg.maxWh      = kMaxWh;

    m_pp = std::make_unique<kernels::ops::CvPostprocessFusedYoloV26ObbPlan>(pcfg);
    if (!m_pp->ok())
    {
        throw std::runtime_error(m_tag + ": rotated postprocess plan construction failed");
    }
}

void YoloV26Obb::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;

    auto logBox = [&](const std::string& nm, int batch, int ch, int h, int w,
                      core::DataType dt, std::size_t bytes, core::MemorySpace sp)
    {
        core::detail::AllocInfo info;
        info.name     = nm.c_str();   // the caller keeps the string alive for this call
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
    logBox(m_tag + ".input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * 3 * W * H * sizeof(float));
    logBox(m_tag + ".input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(static_cast<std::size_t>(B) * m_outC * m_anchors * sizeof(float));
    logBox(m_tag + ".output_src", B, m_outC, 1, m_anchors, core::DataType::Float32,
           m_outputSrc.bytes(), core::MemorySpace::Device);

    const int detStride = m_pp->detStride();
    const std::size_t detBytes =
        static_cast<std::size_t>(B) * m_topK * detStride * sizeof(float);
    const std::size_t cntBytes = static_cast<std::size_t>(B) * sizeof(int);

    m_dets.allocate(detBytes);
    logBox(m_tag + ".dets", B, m_topK, 1, detStride, core::DataType::Float32,
           m_dets.bytes(), core::MemorySpace::Device);

    m_counts.allocate(cntBytes);
    logBox(m_tag + ".counts", B, 1, 1, 1, core::DataType::Int32,
           m_counts.bytes(), core::MemorySpace::Device);

    m_detsHost.allocate(detBytes);
    logBox(m_tag + ".dets_host", B, m_topK, 1, detStride, core::DataType::Float32,
           m_detsHost.bytes(), core::MemorySpace::Host);

    m_countsHost.allocate(cntBytes);
    logBox(m_tag + ".counts_host", B, 1, 1, 1, core::DataType::Int32,
           m_countsHost.bytes(), core::MemorySpace::Host);

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()))
    {
        throw std::runtime_error(m_tag + ": setTensorAddress failed");
    }
}

void YoloV26Obb::init(const core::ModelConfig& cfg)
{
    TRT_LOG_DEBUG(m_tag << "::init: entering");
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    buildPlans();
    allocateBuffers();
    m_obbs.assign(static_cast<std::size_t>(m_cfg.batchSize), {});

    TRT_LOG_INFO(m_tag << ": initialized (batch=" << m_cfg.batchSize
                 << ", outC=" << m_outC
                 << ", anchors=" << m_anchors
                 << ", dst=" << m_cfg.dstW << "x" << m_cfg.dstH << ")");
}

void YoloV26Obb::ensurePreprocessPlan()
{
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
        throw std::runtime_error(m_tag + ": preprocess plan construction failed");
    }
    m_lb = m_pre->info();

    TRT_LOG_DEBUG(m_tag << ": letterbox rebuilt  " << m_srcW << "x" << m_srcH
                 << " -> new " << m_lb.newW << "x" << m_lb.newH
                 << " pad(l/t)=" << m_lb.padLeft << "/" << m_lb.padTop
                 << " ratio=" << m_lb.ratio);
}

void YoloV26Obb::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error(m_tag + ": empty batch");
    }

    m_batch = requireBatchCapacity(*this, batch, m_tag.c_str());
    m_srcH  = batch.views[0].height;
    m_srcW  = batch.views[0].width;
    if (m_srcW <= 0 || m_srcH <= 0)
    {
        throw std::runtime_error(m_tag + ": invalid source size " +
                                 std::to_string(m_srcW) + "x" + std::to_string(m_srcH));
    }

    ensurePreprocessPlan();

    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error(m_tag + ": batch.buffer is null");
    }

    const std::size_t oneImage = 3 * static_cast<std::size_t>(m_srcH) * m_srcW;
    const std::size_t total    = oneImage * batch.views.size();

    if (total > m_inputSrc.bytes())
    {
        m_inputSrc.allocate(total);
    }

    cudaMemcpyAsync(m_inputSrc.data(), batch.buffer->data(), total,
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();
}

void YoloV26Obb::preprocess()
{
    if (!m_pre)
    {
        throw std::runtime_error(m_tag + ": preprocess called before setBatch");
    }
    m_pre->launch(m_stream.get(), m_batch,
                  static_cast<const unsigned char*>(m_inputSrc.data()),
                  m_inputNchw.asFloat());
    if (!m_pre->ok())
    {
        throw std::runtime_error(m_tag + ": preprocess kernel launch failed");
    }
}

void YoloV26Obb::infer()
{
    if (!m_engine->context()->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error(m_tag + ": enqueueV3 failed");
    }
}

void YoloV26Obb::postprocess()
{
    m_obbs.assign(static_cast<std::size_t>(m_batch), {});

    m_pp->launch(m_stream.get(), m_batch, m_outputSrc.asFloat(),
                 static_cast<int*>(m_counts.data()), m_dets.asFloat());
    if (!m_pp->ok())
    {
        throw std::runtime_error(m_tag + ": rotated postprocess kernel launch failed");
    }

    const int         stride   = m_pp->detStride();
    const std::size_t cntBytes = static_cast<std::size_t>(m_batch) * sizeof(int);
    const std::size_t detBytes =
        static_cast<std::size_t>(m_batch) * m_topK * stride * sizeof(float);

    cudaMemcpyAsync(m_countsHost.data(), m_counts.data(), cntBytes,
                    cudaMemcpyDeviceToHost, m_stream.get());
    cudaMemcpyAsync(m_detsHost.data(), m_dets.data(), detBytes,
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    const int*   counts = static_cast<const int*>(m_countsHost.data());
    const float* dets   = m_detsHost.asFloat();

    // ops.scale_boxes(..., xywh=True): centre -= pad then /= gain, w/h only
    // /= gain. No clipping -- an oriented box has no axis-aligned bounds.
    const double gain = m_lb.ratio;

    for (int b = 0; b < m_batch; ++b)
    {
        const int n = std::clamp(counts[b], 0, m_topK);
        const float* row = dets + static_cast<std::size_t>(b) * m_topK * stride;
        auto& out = m_obbs[static_cast<std::size_t>(b)];
        out.clear();
        out.reserve(static_cast<std::size_t>(n));

        for (int i = 0; i < n; ++i)
        {
            const float* d = row + static_cast<std::size_t>(i) * stride;

            ObbDetection o;
            o.cx            = static_cast<float>(m_lb.toOrigX(d[0]));
            o.cy            = static_cast<float>(m_lb.toOrigY(d[1]));
            o.w             = static_cast<float>(static_cast<double>(d[2]) / gain);
            o.h             = static_cast<float>(static_cast<double>(d[3]) / gain);
            o.confidence    = d[4];
            o.label         = static_cast<int>(d[5]);
            o.angle         = d[6];
            out.push_back(o);
        }
    }

    TRT_LOG_DEBUG(m_tag << "::postprocess: batch=" << m_batch
                 << " obbs[0]=" << (m_obbs.empty() ? 0 : m_obbs[0].size()));
}

void YoloV26Obb::reset()
{
    for (auto& v : m_obbs) { v.clear(); }
    m_batch = 0;
}

}  // namespace trt_alpha::obb

TRT_ALPHA_REGISTER_MODEL("yolov26_obb", trt_alpha::obb::YoloV26Obb);
