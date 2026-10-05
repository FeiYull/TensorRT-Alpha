// =============================================================================
//  trt_alpha :: det :: YoloX（实现）
// =============================================================================
#include "yolox.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace trt_alpha::det {
namespace {

//! YOLOX 的 letterbox 几何：padding 在【右边界、下边界】。
//! src2dst: [scale, 0, (scale-1)*0.5; 0, scale, (scale-1)*0.5]
void buildYoloxLetterboxAffine(int srcW, int srcH, int dstW, int dstH,
                               kernels::AffineMat& dst2src)
{
    const float a = static_cast<float>(dstH) / static_cast<float>(srcH);
    const float b = static_cast<float>(dstW) / static_cast<float>(srcW);
    const float scale = std::min(a, b);
    const float tx = (scale - 1.f) * 0.5f;
    const float ty = (scale - 1.f) * 0.5f;
    const float invScale = 1.f / scale;
    dst2src.v0 = invScale;  dst2src.v1 = 0.f;  dst2src.v2 = -tx * invScale;
    dst2src.v3 = 0.f;       dst2src.v4 = invScale; dst2src.v5 = -ty * invScale;
}

}  // namespace

const std::string& YoloX::name() const noexcept
{
    static const std::string kName = "yolox";
    return kName;
}

void YoloX::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    if (m_numClass <= 0) {
        throw std::runtime_error("yolox: num_class must be > 0");
    }

    TRT_LOG_INFO("YoloX: config num_class=" << m_numClass
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK
                 << " scale=" << m_normScale);
}

void YoloX::discoverEngineIo()
{
    const core::TensorDesc* input = nullptr;
    const core::TensorDesc* output = nullptr;
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput && input == nullptr)  { input = &t; }
        if (!t.isInput && output == nullptr){ output = &t; }
    }
    if (input == nullptr || output == nullptr)
    {
        throw std::runtime_error("yolox: engine must have >=1 input and >=1 output");
    }
    m_inputName = input->name;
    m_outputName = output->name;

    // 解析并下发输入形状（秩 / 通道轴 / 格式在这里统一校验）
    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 3)
    {
        throw std::runtime_error("yolox: expect output as [batch, anchors, 5+nc]");
    }
    m_anchors = static_cast<int>(outDims.d[1]);
    m_srcRow  = static_cast<int>(outDims.d[2]);
    if (m_srcRow != 5 + m_numClass)
    {
        throw std::runtime_error(
            "yolox: output channel = " + std::to_string(m_srcRow) +
            " but 5+num_class = " + std::to_string(5 + m_numClass));
    }
}

void YoloX::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;
    const std::size_t dstArea = static_cast<std::size_t>(H) * W;
    const std::size_t oneImageF32 = 3 * dstArea * sizeof(float);

    auto logBox = [&](const char* nm, int batch, int ch, int h, int w,
                      core::DataType dt, std::size_t bytes, core::MemorySpace sp)
    {
        core::detail::AllocInfo info;
        info.name = nm; info.batch = batch; info.channels = ch;
        info.height = h; info.width = w; info.dtype = dt;
        info.bytes = bytes; info.space = sp;
        core::detail::logAllocBox(info);
    };

    m_inputSrc.allocate(static_cast<std::size_t>(B) * 3 * W * H);
    logBox("yolox.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_resizeOut.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolox.resize_out", B, 3, H, W, core::DataType::Float32,
           m_resizeOut.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolox.input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(static_cast<std::size_t>(B) * m_anchors * m_srcRow * sizeof(float));
    logBox("yolox.output_src", B, m_srcRow, 1, m_anchors, core::DataType::Float32,
           m_outputSrc.bytes(), core::MemorySpace::Device);

    m_objectsPerImage = 1 + kernels::kObjectWidth * m_topK;
    m_objects.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolox.objects", B, m_objectsPerImage, 1, 1, core::DataType::Float32,
           m_objects.bytes(), core::MemorySpace::Device);

    m_objectsHost.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolox.objects_host", B, m_objectsPerImage, 1, 1,
           core::DataType::Float32, m_objectsHost.bytes(), core::MemorySpace::Host);

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()))
    {
        throw std::runtime_error("yolox: setTensorAddress failed");
    }
}

void YoloX::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();
    m_detections.assign(static_cast<std::size_t>(m_cfg.batchSize), {});
    TRT_LOG_INFO("YoloX: initialized (batch=" << m_cfg.batchSize
                 << ", anchors=" << m_anchors
                 << ", srcRow=" << m_srcRow << ")");
}

void YoloX::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yolox: empty batch");
    }
    m_batch = static_cast<int>(batch.views.size());
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    buildYoloxLetterboxAffine(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH, m_dst2src);

    const std::size_t oneImage = 3 * static_cast<std::size_t>(m_srcH) * m_srcW;
    const std::size_t total = oneImage * batch.views.size();
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolox: batch.buffer is null");
    }
    if (total > m_inputSrc.bytes())
    {
        m_inputSrc.allocate(total);
    }
    cudaMemcpyAsync(m_inputSrc.data(), batch.buffer->data(), total,
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();
}

void YoloX::preprocess()
{
    kernels::resizeLetterbox(m_stream.get(), m_batch,
                             static_cast<const std::uint8_t*>(m_inputSrc.data()),
                             m_srcW, m_srcH,
                             m_resizeOut.asFloat(),
                             m_cfg.dstW, m_cfg.dstH,
                             m_padValue, m_dst2src);

    // YOLOX: 不转 RGB（swapRB=false），不归一化（scale=1, mean=0, std=1）
    kernels::bgrToNchwNormalized(m_stream.get(), m_batch,
                                 m_resizeOut.asFloat(),
                                 m_inputNchw.asFloat(),
                                 m_cfg.dstW, m_cfg.dstH,
                                 m_normScale, m_normMean, m_normStd,
                                 /*swapRB=*/false);
}

void YoloX::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolox: enqueueV3 failed");
    }
}

void YoloX::postprocess()
{
    m_detections.assign(static_cast<std::size_t>(m_batch), {});

    kernels::YoloDecodeParams p;
    p.batch = m_batch;
    p.numClasses = m_numClass;
    p.topK = m_topK;
    p.confThreshold = m_confThreshold;
    p.iouThreshold = m_iouThreshold;

    cudaMemsetAsync(m_objects.data(), 0,
                    static_cast<std::size_t>(m_objectsPerImage) * m_cfg.batchSize * sizeof(float),
                    m_stream.get());

    kernels::decodeYoloV5Head(m_stream.get(), p, m_outputSrc.asFloat(),
                              m_anchors, m_objects.asFloat());

    kernels::nmsFast(m_stream.get(), p, m_objects.asFloat(), kernels::kObjectWidth);

    cudaMemcpyAsync(m_objectsHost.data(), m_objects.data(),
                    static_cast<std::size_t>(m_objectsPerImage) * m_batch * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    const float* host = m_objectsHost.asFloat();
    for (int b = 0; b < m_batch; ++b)
    {
        const float* row = host + static_cast<std::size_t>(b) * m_objectsPerImage;
        const int count = std::clamp(static_cast<int>(row[0]), 0, m_topK);
        m_detections[static_cast<std::size_t>(b)].clear();
        for (int i = 0; i < count; ++i)
        {
            const float* o = row + 1 + i * kernels::kObjectWidth;
            if (o[6] < 0.5f)
            {
                continue;
            }
            Detection d;
            const float nx = o[0], ny = o[1], nr = o[2], nb = o[3];
            d.left   = m_dst2src.v0 * nx + m_dst2src.v1 * ny + m_dst2src.v2;
            d.top    = m_dst2src.v3 * nx + m_dst2src.v4 * ny + m_dst2src.v5;
            d.right  = m_dst2src.v0 * nr + m_dst2src.v1 * nb + m_dst2src.v2;
            d.bottom = m_dst2src.v3 * nr + m_dst2src.v4 * nb + m_dst2src.v5;
            d.confidence = o[4];
            d.label = static_cast<int>(o[5]);
            m_detections[static_cast<std::size_t>(b)].push_back(d);
        }
    }
}

void YoloX::reset()
{
    for (auto& v : m_detections)
    {
        v.clear();
    }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("yolox", trt_alpha::det::YoloX);