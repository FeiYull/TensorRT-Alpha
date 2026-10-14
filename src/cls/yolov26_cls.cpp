// =============================================================================
//  trt_alpha :: cls :: YoloV26Cls（实现）
// -----------------------------------------------------------------------------
//  管线：
//      setBatch   : 原图 H2D
//      preprocess : CvPreprocessVecClsPlan —— Resize(短边) + CenterCrop +
//                   BGR2RGB + HWC2CHW + /255，单 kernel
//      infer      : enqueueV3
//      postprocess: CvTopKPlan —— top-k，单 kernel；然后 D2H
//
//  为什么不需要别的东西：
//      output0 已经是 softmax 后的概率（实测 sum == 1），也没有 NMS 这一环。
//      ultralytics 的 Results.probs 就是 torch.topk(probs, 5)，所以这里一个
//      topk 算子就是全部后处理。
// =============================================================================
#include "yolov26_cls.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace trt_alpha::cls {

const std::string& YoloV26Cls::name() const noexcept
{
    static const std::string kName = "yolov26_cls";
    return kName;
}

void YoloV26Cls::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    if (m_numClass <= 0)
    {
        throw std::runtime_error("yolov26_cls: num_class must be > 0 (got " +
                                 std::to_string(m_numClass) + ")");
    }
    if (m_topK <= 0)
    {
        throw std::runtime_error("yolov26_cls: top_k must be > 0 (got " +
                                 std::to_string(m_topK) + ")");
    }

    TRT_LOG_INFO("YoloV26Cls: config num_class=" << m_numClass
                 << " top_k=" << m_topK
                 << " scale=" << m_normScale
                 << " pad_value=" << m_padValue);
}

void YoloV26Cls::discoverEngineIo()
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
        throw std::runtime_error("yolov26_cls: engine must have >=1 input and >=1 output");
    }
    m_inputName  = input->name;
    m_outputName = output->name;

    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    const nvinfer1::Dims d = m_engine->contextShape(m_outputName);
    if (d.nbDims != 2)
    {
        throw std::runtime_error(
            "yolov26_cls: expect output0 as [B, num_class] (got nbDims=" +
            std::to_string(d.nbDims) + ")");
    }
    if (d.d[0] > 0 && d.d[0] != m_cfg.batchSize)
    {
        throw std::runtime_error(
            "yolov26_cls: output0 batch = " + std::to_string(d.d[0]) +
            " but input batch = " + std::to_string(m_cfg.batchSize) +
            " (engine profile is the single source of truth; check 'batch_size')");
    }

    m_scores = static_cast<int>(d.d[1]);
    if (m_scores <= 0)
    {
        throw std::runtime_error("yolov26_cls: output0 num_class must be > 0");
    }
    if (m_scores != m_numClass)
    {
        throw std::runtime_error(
            "yolov26_cls: output0 length = " + std::to_string(m_scores) +
            " but num_class = " + std::to_string(m_numClass) +
            " (check 'num_class' in the INI)");
    }
    if (m_topK > m_scores)
    {
        throw std::runtime_error(
            "yolov26_cls: top_k = " + std::to_string(m_topK) +
            " exceeds num_class = " + std::to_string(m_scores));
    }
}

void YoloV26Cls::allocateBuffers()
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
    logBox("yolov26_cls.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * 3 * W * H * sizeof(float));
    logBox("yolov26_cls.input_nchw", B, 3, W, H, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_logits.allocate(static_cast<std::size_t>(B) * m_scores * sizeof(float));
    logBox("yolov26_cls.logits", B, 1, 1, m_scores, core::DataType::Float32,
           m_logits.bytes(), core::MemorySpace::Device);

    const std::size_t topkBytes =
        static_cast<std::size_t>(B) * 2 * m_topK * sizeof(float);
    m_topk.allocate(topkBytes);
    logBox("yolov26_cls.topk", B, 2, 1, m_topK, core::DataType::Float32,
           m_topk.bytes(), core::MemorySpace::Device);

    m_topkHost.allocate(topkBytes);
    logBox("yolov26_cls.topk_host", B, 2, 1, m_topK, core::DataType::Float32,
           m_topkHost.bytes(), core::MemorySpace::Host);

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_logits.data()))
    {
        throw std::runtime_error("yolov26_cls: setTensorAddress failed");
    }
}

void YoloV26Cls::init(const core::ModelConfig& cfg)
{
    TRT_LOG_DEBUG("YoloV26Cls::init: entering");
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();
    m_topkPlan = std::make_unique<kernels::ops::CvTopKPlan>(
        m_scores, m_topK, m_cfg.batchSize);
    if (!m_topkPlan->ok())
    {
        throw std::runtime_error("yolov26_cls: topk plan construction failed");
    }

    TRT_LOG_INFO("YoloV26Cls: initialized (batch=" << m_cfg.batchSize
                 << ", crop=" << m_cfg.dstW << "x" << m_cfg.dstH
                 << ", num_class=" << m_scores << ", top_k=" << m_topK << ")");
}

void YoloV26Cls::ensurePlans()
{
    if (m_pre && m_pre->matches(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH))
    {
        return;
    }

    m_pre = std::make_unique<kernels::ops::CvPreprocessVecClsPlan>(
        kernels::ops::CvPreprocessVecClsPlan::byDivisor(
            m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH, m_normScale));
    if (!m_pre->ok())
    {
        throw std::runtime_error("yolov26_cls: preprocess plan construction failed");
    }

    const kernels::ops::PreprocessPilClsConfig& g = m_pre->info();
    TRT_LOG_DEBUG("YoloV26Cls: preprocess rebuilt  src=" << m_srcW << "x" << m_srcH
                 << " -> resize " << g.newW << "x" << g.newH
                 << " -> crop[" << g.cropTop << ":" << (g.cropTop + g.outH)
                 << "," << g.cropLeft << ":" << (g.cropLeft + g.outW) << "]"
                 << " needH=" << g.needH << " needV=" << g.needV);
}

void YoloV26Cls::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yolov26_cls: empty batch");
    }

    m_batch = requireBatchCapacity(*this, batch, "yolov26_cls");
    m_srcH  = batch.views[0].height;
    m_srcW  = batch.views[0].width;
    if (m_srcW <= 0 || m_srcH <= 0)
    {
        throw std::runtime_error("yolov26_cls: invalid source size " +
                                 std::to_string(m_srcW) + "x" + std::to_string(m_srcH));
    }

    ensurePlans();

    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolov26_cls: batch.buffer is null");
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

void YoloV26Cls::preprocess()
{
    if (!m_pre)
    {
        throw std::runtime_error("yolov26_cls: preprocess called before setBatch");
    }
    m_pre->launch(m_stream.get(), m_batch,
                  static_cast<const unsigned char*>(m_inputSrc.data()),
                  m_inputNchw.asFloat());
    if (!m_pre->ok())
    {
        throw std::runtime_error("yolov26_cls: preprocess kernel launch failed");
    }
}

void YoloV26Cls::infer()
{
    if (!m_engine->context()->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolov26_cls: enqueueV3 failed");
    }
}

void YoloV26Cls::postprocess()
{
    m_classifications.assign(static_cast<std::size_t>(m_batch), {});

    m_topkPlan->launch(m_stream.get(), m_batch, m_logits.asFloat(),
                       m_topk.asFloat());
    if (!m_topkPlan->ok())
    {
        throw std::runtime_error("yolov26_cls: topk kernel launch failed");
    }

    const std::size_t bytes =
        static_cast<std::size_t>(m_batch) * m_topkPlan->outElems() * sizeof(float);
    cudaMemcpyAsync(m_topkHost.data(), m_topk.data(), bytes,
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    const float* src = m_topkHost.asFloat();

    for (int b = 0; b < m_batch; ++b)
    {
        const float* row = src + static_cast<std::size_t>(b) * m_topkPlan->outElems();
        auto& out = m_classifications[static_cast<std::size_t>(b)];
        out.clear();
        out.reserve(static_cast<std::size_t>(m_topK));

        for (int i = 0; i < m_topK; ++i)
        {
            ClassScore cs;
            cs.label = static_cast<int>(row[2 * i]);
            cs.score = row[2 * i + 1];
            out.push_back(cs);
        }
    }

    if (m_batch > 0 && !m_classifications[0].empty())
    {
        TRT_LOG_DEBUG("YoloV26Cls::postprocess: batch[0] top1 = class "
                     << m_classifications[0][0].label
                     << " (" << m_classifications[0][0].score << ")");
    }
}

void YoloV26Cls::reset()
{
    for (auto& v : m_classifications)
    {
        v.clear();
    }
    m_batch = 0;
}

}  // namespace trt_alpha::cls

TRT_ALPHA_REGISTER_MODEL("yolov26_cls", trt_alpha::cls::YoloV26Cls);
