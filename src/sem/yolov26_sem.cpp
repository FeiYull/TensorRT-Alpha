// =============================================================================
//  trt_alpha :: sem :: YoloV26Sem（实现）
// -----------------------------------------------------------------------------
//  管线：
//      setBatch   : 原图 H2D
//      preprocess : CvPreprocessVecPlan —— letterbox 640x640 + BGR2RGB +
//                   HWC2CHW + /255（与检测逐位相同）
//      infer      : enqueueV3
//      postprocess: CvScaleMaskPlan<unsigned char>(Nearest) —— 裁灰边 + 最近邻
//                   回原图，单 kernel；然后 D2H 成 SemanticMask
//
//  为什么是 Nearest 而不是 Bilinear：
//      output0 已经是 argmax 后的类别图（uint8 0..18）。对类别图做双线性等于对
//      类别 id 求加权平均，会凭空造出 "7.4 类" 这种无意义的值 —— 所以算子本身
//      在 Bilinear + 非 float 时就直接拒绝（见 CvScaleMaskPlan 构造函数），
//      这里显式选 Nearest，与 ultralytics scale_masks(mode="nearest") 同口径。
// =============================================================================
#include "yolov26_sem.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trt_alpha::sem {

const std::string& YoloV26Sem::name() const noexcept
{
    static const std::string kName = "yolov26_sem";
    return kName;
}

void YoloV26Sem::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);
    TRT_LOG_INFO("YoloV26Sem: config scale=" << m_normScale
                 << " pad_value=" << m_padValue);
}

void YoloV26Sem::discoverEngineIo()
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
        throw std::runtime_error("yolov26_sem: engine must have >=1 input and >=1 output");
    }
    m_inputName  = input->name;
    m_outputName = output->name;

    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    // ---- 这是全仓唯一一个非 fp32 的引擎输出，必须显式核对 ----
    // TRT 不会替我们检查：按 float 去绑一个 UINT8 张量不会报错，只会静默读错
    // 内存（每 4 个类别像素当成 1 个 float 解释）。所以这里连 dtype 和 format
    // 一起卡死。
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.name != m_outputName)
        {
            continue;
        }
        if (t.dtype != core::DataType::UInt8)
        {
            throw std::runtime_error(
                "yolov26_sem: output0 must be UINT8 (an argmax'ed class map), but the "
                "engine declares a different dtype; binding it as float would silently "
                "read the wrong memory");
        }
        if (t.format != nvinfer1::TensorFormat::kLINEAR)
        {
            throw std::runtime_error(
                "yolov26_sem: output0 must be a LINEAR tensor, but the engine declares " +
                t.formatDesc + "; the row-major byte indexing below would be wrong");
        }
    }

    const nvinfer1::Dims d = m_engine->contextShape(m_outputName);
    if (d.nbDims == 3 && d.d[0] > 0 && d.d[0] != m_cfg.batchSize)
    {
        throw std::runtime_error(
            "yolov26_sem: output0 batch = " + std::to_string(d.d[0]) +
            " but input batch = " + std::to_string(m_cfg.batchSize) +
            " (engine profile is the single source of truth; check 'batch_size')");
    }

    if (d.nbDims == 3)
    {
        // [B, H, W]
        m_outH = static_cast<int>(d.d[1]);
        m_outW = static_cast<int>(d.d[2]);
    }
    else if (d.nbDims == 4 && d.d[1] == 1)
    {
        // [B, 1, H, W] -- accepted too, so an engine exported with a keepdim
        // argmax does not need a graph change.
        m_outH = static_cast<int>(d.d[2]);
        m_outW = static_cast<int>(d.d[3]);
    }
    else
    {
        throw std::runtime_error("yolov26_sem: expect output0 as [B,H,W] or [B,1,H,W]");
    }

    if (m_outH <= 0 || m_outW <= 0)
    {
        throw std::runtime_error("yolov26_sem: output0 spatial dims must be > 0");
    }
}

void YoloV26Sem::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;

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
    logBox("yolov26_sem.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * 3 * W * H * sizeof(float));
    logBox("yolov26_sem.input_nchw", B, 3, W, H, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    const std::size_t netBytes = static_cast<std::size_t>(B) * m_outH * m_outW;
    m_clsNet.allocate(netBytes);
    logBox("yolov26_sem.class_net", B, 1, m_outH, m_outW, core::DataType::UInt8,
           m_clsNet.bytes(), core::MemorySpace::Device);

    // 目标尺寸要等 setBatch 拿到原图，先按网络尺寸预留
    m_clsOrig.allocate(netBytes);
    m_clsHost.allocate(netBytes);

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    // ---- 用原始字节指针绑 UINT8 输出，绝不能 asFloat() ----
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_clsNet.data()))
    {
        throw std::runtime_error("yolov26_sem: setTensorAddress failed");
    }
}

void YoloV26Sem::init(const core::ModelConfig& cfg)
{
    TRT_LOG_DEBUG("YoloV26Sem::init: entering");
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();

    TRT_LOG_INFO("YoloV26Sem: initialized (batch=" << m_cfg.batchSize
                 << ", net=" << m_cfg.dstW << "x" << m_cfg.dstH
                 << ", class_map=" << m_outW << "x" << m_outH << " u8)");
}

void YoloV26Sem::ensurePlans()
{
    if (m_pre && m_pre->matches(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH))
    {
        return;
    }

    m_pre = std::make_unique<kernels::ops::CvPreprocessVecPlan>(
        kernels::ops::CvPreprocessVecPlan::byDivisor(
            m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH,
            static_cast<double>(m_padValue), m_normScale));
    if (!m_pre->ok())
    {
        throw std::runtime_error("yolov26_sem: preprocess plan construction failed");
    }

    m_sm = std::make_unique<kernels::ops::CvScaleMaskPlan<unsigned char>>(
        m_srcW, m_srcH, m_outW, m_outH, kernels::ops::MaskInterp::Nearest);
    if (!m_sm->ok())
    {
        throw std::runtime_error("yolov26_sem: scale_mask plan construction failed");
    }

    const std::size_t need = static_cast<std::size_t>(m_cfg.batchSize) * m_sm->outElems();
    if (need > m_clsOrig.bytes())
    {
        m_clsOrig.allocate(need);
        m_clsHost.allocate(need);
    }

    TRT_LOG_DEBUG("YoloV26Sem: plans rebuilt  " << m_srcW << "x" << m_srcH
                 << " crop=[" << m_sm->info().cropTop << ":"
                 << (m_sm->info().cropTop + m_sm->info().newH) << ","
                 << m_sm->info().cropLeft << ":"
                 << (m_sm->info().cropLeft + m_sm->info().newW) << "]");
}

void YoloV26Sem::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yolov26_sem: empty batch");
    }

    m_batch = requireBatchCapacity(*this, batch, "yolov26_sem");
    m_srcH  = batch.views[0].height;
    m_srcW  = batch.views[0].width;
    if (m_srcW <= 0 || m_srcH <= 0)
    {
        throw std::runtime_error("yolov26_sem: invalid source size " +
                                 std::to_string(m_srcW) + "x" + std::to_string(m_srcH));
    }

    ensurePlans();

    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolov26_sem: batch.buffer is null");
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

void YoloV26Sem::preprocess()
{
    if (!m_pre)
    {
        throw std::runtime_error("yolov26_sem: preprocess called before setBatch");
    }
    m_pre->launch(m_stream.get(), m_batch,
                  static_cast<const unsigned char*>(m_inputSrc.data()),
                  m_inputNchw.asFloat());
    if (!m_pre->ok())
    {
        throw std::runtime_error("yolov26_sem: preprocess kernel launch failed");
    }
}

void YoloV26Sem::infer()
{
    if (!m_engine->context()->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolov26_sem: enqueueV3 failed");
    }
}

void YoloV26Sem::postprocess()
{
    m_semantics.assign(static_cast<std::size_t>(m_batch), {});

    m_sm->launch(m_stream.get(), m_batch,
                 static_cast<const unsigned char*>(m_clsNet.data()),
                 static_cast<unsigned char*>(m_clsOrig.data()),
                 /*channels=*/1);
    if (!m_sm->ok())
    {
        throw std::runtime_error("yolov26_sem: scale_mask kernel launch failed");
    }

    const std::size_t elems = static_cast<std::size_t>(m_batch) * m_sm->outElems();
    cudaMemcpyAsync(m_clsHost.data(), m_clsOrig.data(), elems,
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    const auto* src       = static_cast<const std::uint8_t*>(m_clsHost.data());
    const std::size_t per = m_sm->outElems();

    for (int b = 0; b < m_batch; ++b)
    {
        SemanticMask& sm = m_semantics[static_cast<std::size_t>(b)];
        sm.width  = m_sm->outW();
        sm.height = m_sm->outH();
        sm.data.assign(src + static_cast<std::size_t>(b) * per,
                       src + static_cast<std::size_t>(b + 1) * per);
    }

    TRT_LOG_DEBUG("YoloV26Sem::postprocess: " << m_batch << " mask(s) "
                 << m_sm->outW() << "x" << m_sm->outH());
}

void YoloV26Sem::reset()
{
    for (auto& s : m_semantics)
    {
        s.data.clear();
    }
    m_batch = 0;
}

}  // namespace trt_alpha::sem

TRT_ALPHA_REGISTER_MODEL("yolov26_sem", trt_alpha::sem::YoloV26Sem);
