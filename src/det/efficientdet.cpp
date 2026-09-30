// =============================================================================
//  trt_alpha :: det :: EfficientDet（实现）
// =============================================================================
#include "efficientdet.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/kernels/cast.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_alpha::det {
namespace {

void buildLetterboxAffine(int srcW, int srcH, int dstW, int dstH,
                          kernels::AffineMat& dst2src)
{
    const float a = static_cast<float>(dstH) / static_cast<float>(srcH);
    const float b = static_cast<float>(dstW) / static_cast<float>(srcW);
    const float scale = std::min(a, b);
    const float tx = (-scale * srcW + dstW + scale - 1.f) * 0.5f;
    const float ty = (-scale * srcH + dstH + scale - 1.f) * 0.5f;
    const float invScale = 1.f / scale;
    dst2src.v0 = invScale;  dst2src.v1 = 0.f;  dst2src.v2 = -tx * invScale;
    dst2src.v3 = 0.f;       dst2src.v4 = invScale; dst2src.v5 = -ty * invScale;
}

}  // namespace

const std::string& EfficientDet::name() const noexcept
{
    static const std::string kName = "efficientdet";
    return kName;
}

void EfficientDet::loadConfig(const core::ModelConfig& cfg)
{
    m_cfg = cfg;
    m_numClass = cfg.getInt("num_class", 91);
    m_confThreshold = cfg.getFloat("conf_thresh", 0.45f);
    m_padValue = cfg.getFloat("pad_value", 114.f);
    m_topK = 100;

    TRT_LOG_INFO("EfficientDet: config num_class=" << m_numClass
                 << " conf=" << m_confThreshold
                 << " topK=" << m_topK);
}

void EfficientDet::discoverEngineIo()
{
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput)                     { m_inputName = t.name;   continue; }
        if (t.name == "num_detections")    { m_numName = t.name;     continue; }
        if (t.name == "detection_boxes")   { m_boxesName = t.name;   continue; }
        if (t.name == "detection_scores")  { m_scoresName = t.name;  continue; }
        if (t.name == "detection_classes") { m_classesName = t.name; continue; }
    }
    if (m_inputName.empty() || m_numName.empty() || m_boxesName.empty() ||
        m_scoresName.empty() || m_classesName.empty())
    {
        throw std::runtime_error("efficientdet: missing expected I/O tensors");
    }

    const core::TensorDesc* in = m_engine->find(m_inputName);
    if (in == nullptr || in->shape.nbDims != 4)
    {
        throw std::runtime_error("efficientdet: expect input as [B, H, W, 3]");
    }
    if (in->shape.d[3] != 3)
    {
        throw std::runtime_error("efficientdet: input channel dim != 3 (expect NHWC)");
    }

    m_engine->setInputShape(m_inputName, nvinfer1::Dims4(
        m_cfg.batchSize, m_cfg.dstH, m_cfg.dstW, 3));

    TRT_LOG_INFO("EfficientDet: input '" << m_inputName
                 << "' set to [" << m_cfg.batchSize << ", "
                 << m_cfg.dstH << ", " << m_cfg.dstW << ", 3]");
}

void EfficientDet::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;

    auto logBox = [&](const char* nm, int batch, int ch, int h, int w,
                      core::DataType dt, std::size_t bytes, core::MemorySpace sp)
    {
        core::detail::AllocInfo info;
        info.name = nm; info.batch = batch; info.channels = ch;
        info.height = h; info.width = w; info.dtype = dt;
        info.bytes = bytes; info.space = sp;
        core::detail::logAllocBox(info);
    };

    // 输入 float32 NHWC
    m_inputSrc.allocate(static_cast<std::size_t>(B) * H * W * 3 * sizeof(float));
    logBox("efficientdet.input_src", B, 3, H, W, core::DataType::Float32,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    // H2D 临时 uint8 buffer
    m_inputU8.allocate(static_cast<std::size_t>(B) * H * W * 3);
    logBox("efficientdet.input_u8", B, 3, H, W, core::DataType::UInt8,
           m_inputU8.bytes(), core::MemorySpace::Device);

    m_inputRgb.allocate(static_cast<std::size_t>(B) * H * W * 3 * sizeof(float));
    logBox("efficientdet.input_rgb", B, 3, H, W, core::DataType::Float32,
           m_inputRgb.bytes(), core::MemorySpace::Device);

    m_outputNum.allocate(static_cast<std::size_t>(B) * sizeof(std::int32_t));
    logBox("efficientdet.num", B, 1, 1, 1, core::DataType::Int32,
           m_outputNum.bytes(), core::MemorySpace::Device);

    m_outputBoxes.allocate(static_cast<std::size_t>(B) * m_topK * 4 * sizeof(float));
    logBox("efficientdet.boxes", B, 4, m_topK, 1, core::DataType::Float32,
           m_outputBoxes.bytes(), core::MemorySpace::Device);

    m_outputScores.allocate(static_cast<std::size_t>(B) * m_topK * sizeof(float));
    logBox("efficientdet.scores", B, 1, m_topK, 1, core::DataType::Float32,
           m_outputScores.bytes(), core::MemorySpace::Device);

    m_outputClasses.allocate(static_cast<std::size_t>(B) * m_topK * sizeof(std::int32_t));
    logBox("efficientdet.classes", B, 1, m_topK, 1, core::DataType::Int32,
           m_outputClasses.bytes(), core::MemorySpace::Device);

    m_hostNum.allocate(static_cast<std::size_t>(B) * sizeof(std::int32_t));
    m_hostBoxes.allocate(static_cast<std::size_t>(B) * m_topK * 4 * sizeof(float));
    m_hostScores.allocate(static_cast<std::size_t>(B) * m_topK * sizeof(float));
    m_hostClasses.allocate(static_cast<std::size_t>(B) * m_topK * sizeof(std::int32_t));

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputRgb.data()) ||
        !ctx->setTensorAddress(m_numName.c_str(), m_outputNum.data()) ||
        !ctx->setTensorAddress(m_boxesName.c_str(), m_outputBoxes.data()) ||
        !ctx->setTensorAddress(m_scoresName.c_str(), m_outputScores.data()) ||
        !ctx->setTensorAddress(m_classesName.c_str(), m_outputClasses.data()))
    {
        throw std::runtime_error("efficientdet: setTensorAddress failed");
    }
}

void EfficientDet::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();
    m_detections.assign(static_cast<std::size_t>(m_cfg.batchSize), {});
    TRT_LOG_INFO("EfficientDet: initialized (batch=" << m_cfg.batchSize
                 << ", dst=" << m_cfg.dstW << "x" << m_cfg.dstH << ")");
}

void EfficientDet::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("efficientdet: empty batch");
    }
    m_batch = static_cast<int>(batch.views.size());
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    buildLetterboxAffine(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH, m_dst2src);

    // batch.buffer 是 uint8 Host 连续内存
    const std::size_t oneImageU8 = static_cast<std::size_t>(m_srcH) * m_srcW * 3;
    const std::size_t totalU8 = oneImageU8 * batch.views.size();
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("efficientdet: batch.buffer is null");
    }

    // H2D uint8 -> GPU kernel -> float
    if (m_inputU8.bytes() < totalU8)
    {
        m_inputU8.allocate(totalU8);
    }
    cudaMemcpyAsync(m_inputU8.data(), batch.buffer->data(), totalU8,
                    cudaMemcpyHostToDevice, m_stream.get());
    kernels::u8ToF32(m_stream.get(),
                     static_cast<const std::uint8_t*>(m_inputU8.data()),
                     m_inputSrc.asFloat(),
                     totalU8);
    m_stream.synchronize();
}

void EfficientDet::preprocess()
{
    // 1) letterbox resize：float BGR HWC -> float BGR HWC
    kernels::resizeLetterbox(m_stream.get(), m_batch,
                             m_inputSrc.asFloat(),
                             m_srcW, m_srcH,
                             m_inputRgb.asFloat(),
                             m_cfg.dstW, m_cfg.dstH,
                             m_padValue, m_dst2src);

    // 2) BGR -> RGB（in-place，保持 HWC）
    kernels::bgrToRgbHwc(m_stream.get(), m_batch,
                         m_inputRgb.asFloat(),
                         m_cfg.dstW, m_cfg.dstH);
}

void EfficientDet::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("efficientdet: enqueueV3 failed");
    }
}

void EfficientDet::postprocess()
{
    m_detections.assign(static_cast<std::size_t>(m_batch), {});

    cudaMemcpyAsync(m_hostNum.data(), m_outputNum.data(),
                    static_cast<std::size_t>(m_batch) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost, m_stream.get());
    cudaMemcpyAsync(m_hostBoxes.data(), m_outputBoxes.data(),
                    static_cast<std::size_t>(m_batch) * m_topK * 4 * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream.get());
    cudaMemcpyAsync(m_hostScores.data(), m_outputScores.data(),
                    static_cast<std::size_t>(m_batch) * m_topK * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream.get());
    cudaMemcpyAsync(m_hostClasses.data(), m_outputClasses.data(),
                    static_cast<std::size_t>(m_batch) * m_topK * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    const std::int32_t* numHost     = static_cast<const std::int32_t*>(m_hostNum.data());
    const float*        boxesHost   = static_cast<const float*>(m_hostBoxes.data());
    const float*        scoresHost  = static_cast<const float*>(m_hostScores.data());
    const std::int32_t* classesHost = static_cast<const std::int32_t*>(m_hostClasses.data());

    for (int b = 0; b < m_batch; ++b)
    {
        const int count = std::min(static_cast<int>(numHost[b]), m_topK);
        for (int i = 0; i < count; ++i)
        {
            const float y1 = boxesHost[(b * m_topK + i) * 4 + 0];
            const float x1 = boxesHost[(b * m_topK + i) * 4 + 1];
            const float y2 = boxesHost[(b * m_topK + i) * 4 + 2];
            const float x2 = boxesHost[(b * m_topK + i) * 4 + 3];
            const float score = scoresHost[b * m_topK + i];
            if (score < m_confThreshold) { continue; }
            const int cls = classesHost[b * m_topK + i];

            Detection d;
            d.left   = m_dst2src.v0 * x1 + m_dst2src.v1 * y1 + m_dst2src.v2;
            d.top    = m_dst2src.v3 * x1 + m_dst2src.v4 * y1 + m_dst2src.v5;
            d.right  = m_dst2src.v0 * x2 + m_dst2src.v1 * y2 + m_dst2src.v2;
            d.bottom = m_dst2src.v3 * x2 + m_dst2src.v4 * y2 + m_dst2src.v5;
            d.confidence = score;
            d.label = cls;
            m_detections[static_cast<std::size_t>(b)].push_back(d);
        }
    }
}

void EfficientDet::reset()
{
    for (auto& v : m_detections) { v.clear(); }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("efficientdet", trt_alpha::det::EfficientDet);