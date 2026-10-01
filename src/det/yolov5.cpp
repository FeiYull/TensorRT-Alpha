#include "yolov5.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

const std::string& YoloV5::name() const noexcept
{
    static const std::string kName = "yolov5";
    return kName;
}

void YoloV5::loadConfig(const core::ModelConfig& cfg)
{
    m_cfg = cfg;
    m_numClass = cfg.getInt("num_class", 80);
    if (m_numClass <= 0)
    {
        throw std::runtime_error("yolov5: num_class must be > 0");
    }
    m_confThreshold = cfg.getFloat("conf_thresh", 0.25f);
    m_iouThreshold = cfg.getFloat("iou_thresh", 0.45f);
    m_topK = cfg.getInt("top_k", 300);
    m_normScale = cfg.getFloat("norm_scale", 255.f);
    m_padValue = cfg.getFloat("pad_value", 114.f);

    // mean / std 同 yolov8
    {
        const std::string meanStr = cfg.getString("mean", "");
        if (!meanStr.empty())
        {
            float v[3];
            if (std::sscanf(meanStr.c_str(), "%f,%f,%f", &v[0], &v[1], &v[2]) == 3)
            {
                m_normMean[0] = v[0]; m_normMean[1] = v[1]; m_normMean[2] = v[2];
            }
        }
        const std::string stdStr = cfg.getString("std", "");
        if (!stdStr.empty())
        {
            float v[3];
            if (std::sscanf(stdStr.c_str(), "%f,%f,%f", &v[0], &v[1], &v[2]) == 3)
            {
                m_normStd[0] = v[0]; m_normStd[1] = v[1]; m_normStd[2] = v[2];
            }
        }
    }
    TRT_LOG_INFO("YoloV5: config num_class=" << m_numClass
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK);
}

void YoloV5::discoverEngineIo()
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
        throw std::runtime_error("yolov5: engine must have >=1 input and >=1 output");
    }
    m_inputName = input->name;
    m_outputName = output->name;

    m_engine->setInputShape(m_inputName, nvinfer1::Dims4(m_cfg.batchSize, 3,
                                                          m_cfg.dstH, m_cfg.dstW));

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 3)
    {
        throw std::runtime_error("yolov5: expect output0 as [batch, anchors, 5+nc]");
    }
    m_anchors = static_cast<int>(outDims.d[1]);
    m_srcRow  = static_cast<int>(outDims.d[2]);
    if (m_srcRow != 5 + m_numClass)
    {
        throw std::runtime_error(
            "yolov5: output channel = " + std::to_string(m_srcRow) +
            " but 5+num_class = " + std::to_string(5 + m_numClass) +
            " (check 'num_class' in INI)");
    }
    if (outDims.d[0] < m_cfg.batchSize)
    {
        throw std::runtime_error("yolov5: engine max batch < config batch_size");
    }
}

void YoloV5::allocateBuffers()
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
    logBox("yolov5.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_resizeOut.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolov5.resize_out", B, 3, H, W, core::DataType::Float32,
           m_resizeOut.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolov5.input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(static_cast<std::size_t>(B) * m_anchors * m_srcRow * sizeof(float));
    logBox("yolov5.output_src", B, m_srcRow, 1, m_anchors, core::DataType::Float32,
           m_outputSrc.bytes(), core::MemorySpace::Device);

    m_objectsPerImage = 1 + kernels::kObjectWidth * m_topK;
    m_objects.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolov5.objects", B, m_objectsPerImage, 1, 1, core::DataType::Float32,
           m_objects.bytes(), core::MemorySpace::Device);

    m_objectsHost.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolov5.objects_host", B, m_objectsPerImage, 1, 1,
           core::DataType::Float32, m_objectsHost.bytes(), core::MemorySpace::Host);

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()))
    {
        throw std::runtime_error("yolov5: setTensorAddress failed");
    }
}

void YoloV5::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();
    m_detections.assign(static_cast<std::size_t>(m_cfg.batchSize), {});
    TRT_LOG_INFO("YoloV5: initialized (batch=" << m_cfg.batchSize
                 << ", anchors=" << m_anchors
                 << ", srcRow=" << m_srcRow << ")");
}

void YoloV5::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yolov5: empty batch");
    }
    m_batch = static_cast<int>(batch.views.size());
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    buildLetterboxAffine(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH, m_dst2src);

    const std::size_t oneImage = 3 * static_cast<std::size_t>(m_srcH) * m_srcW;
    const std::size_t total = oneImage * batch.views.size();
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolov5: batch.buffer is null");
    }
    if (total > m_inputSrc.bytes())
    {
        m_inputSrc.allocate(total);
    }
    cudaMemcpyAsync(m_inputSrc.data(), batch.buffer->data(), total,
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();
}

void YoloV5::preprocess()
{
    kernels::resizeLetterbox(m_stream.get(), m_batch,
                             static_cast<const std::uint8_t*>(m_inputSrc.data()),
                             m_srcW, m_srcH,
                             m_resizeOut.asFloat(),
                             m_cfg.dstW, m_cfg.dstH,
                             m_padValue, m_dst2src);

    kernels::bgrToNchwNormalized(m_stream.get(), m_batch,
                                 m_resizeOut.asFloat(),
                                 m_inputNchw.asFloat(),
                                 m_cfg.dstW, m_cfg.dstH,
                                 m_normScale, m_normMean, m_normStd);
}

void YoloV5::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolov5: enqueueV3 failed");
    }
}

void YoloV5::postprocess()
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

    // 不 transpose：直接吃引擎原始输出
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

void YoloV5::reset()
{
    for (auto& v : m_detections)
    {
        v.clear();
    }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("yolov5", trt_alpha::det::YoloV5);