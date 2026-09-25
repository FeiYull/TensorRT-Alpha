// =============================================================================
//  trt_alpha :: det :: YoloV8（实现）
// =============================================================================
#include "yolov8.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"   // （TRT_ALPHA_REGISTER_MODEL 宏）

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trt_alpha::det {
namespace {

//! 计算 letterbox 仿射矩阵（网络输入坐标 → 源图坐标）。
//! 与 ultralytics / TensorRT-Alpha 的像素中心约定一致。
void buildLetterboxAffine(int srcW, int srcH, int dstW, int dstH,
                          kernels::AffineMat& dst2src)
{
    const float a = static_cast<float>(dstH) / static_cast<float>(srcH);
    const float b = static_cast<float>(dstW) / static_cast<float>(srcW);
    const float scale = std::min(a, b);

    const float tx = (-scale * srcW + dstW + scale - 1.f) * 0.5f;
    const float ty = (-scale * srcH + dstH + scale - 1.f) * 0.5f;

    // src -> dst: [scale, 0, tx; 0, scale, ty]
    // dst -> src: [1/scale, 0, -tx/scale; 0, 1/scale, -ty/scale]
    const float invScale = 1.f / scale;
    dst2src.v0 = invScale;  dst2src.v1 = 0.f;  dst2src.v2 = -tx * invScale;
    dst2src.v3 = 0.f;       dst2src.v4 = invScale; dst2src.v5 = -ty * invScale;
}

//! 逐元素 sigmoid（备用，暂时不用）。
[[maybe_unused]] float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }

}  // namespace

const std::string& YoloV8::name() const noexcept
{
    static const std::string kName = "yolov8";
    return kName;
}

void YoloV8::loadConfig(const core::ModelConfig& cfg)
{
    m_cfg = cfg;

    m_numClass = cfg.getInt("num_class", 80);
    if (m_numClass <= 0)
    {
        throw std::runtime_error("yolov8: num_class must be > 0 (got " +
                                 std::to_string(m_numClass) + ")");
    }
    m_confThreshold = cfg.getFloat("conf_thresh", 0.25f);
    m_iouThreshold = cfg.getFloat("iou_thresh", 0.45f);
    m_topK = cfg.getInt("top_k", 300);

    // 归一化参数：mean / std 用 "a,b,c" 逗号分隔
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

    m_normScale = cfg.getFloat("norm_scale", 255.f);
    m_padValue = cfg.getFloat("pad_value", 114.f);

    TRT_LOG_INFO("YoloV8: config num_class=" << m_numClass
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK);
}

void YoloV8::discoverEngineIo()
{
    // 按名字解析（不依赖 binding 顺序）
    const core::TensorDesc* input = nullptr;
    const core::TensorDesc* output = nullptr;
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput && input == nullptr)  { input = &t; }
        if (!t.isInput && output == nullptr){ output = &t; }
    }
    if (input == nullptr || output == nullptr)
    {
        throw std::runtime_error("yolov8: engine must have >=1 input and >=1 output");
    }
    m_inputName = input->name;
    m_outputName = output->name;

    // 设置动态输入形状（静态引擎自动跳过）
    m_engine->setInputShape(m_inputName, nvinfer1::Dims4(m_cfg.batchSize, 3,
                                                          m_cfg.dstH, m_cfg.dstW));

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 3)
    {
        throw std::runtime_error("yolov8: expect output0 as [batch, 4+nc, N]");
    }
    m_srcRow = static_cast<int>(outDims.d[1]);
    m_anchors = static_cast<int>(outDims.d[2]);
    if (m_srcRow != 4 + m_numClass)
    {
        throw std::runtime_error(
            "yolov8: output0 channel = " + std::to_string(m_srcRow) +
            " but 4+num_class = " + std::to_string(4 + m_numClass) +
            " (check 'num_class' in INI)");
    }
    if (outDims.d[0] < m_cfg.batchSize)
    {
        throw std::runtime_error(
            "yolov8: engine max batch " + std::to_string(outDims.d[0]) +
            " < config batch_size " + std::to_string(m_cfg.batchSize));
    }
}

void YoloV8::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;

    const std::size_t dstArea = static_cast<std::size_t>(H) * W;
    const std::size_t oneImageF32 = 3 * dstArea * sizeof(float);

    // --- Device buffer 分配，每个打框图 log ---
    auto logBox = [&](const char* nm, int batch, int ch, int h, int w,
                      core::DataType dt, std::size_t bytes, core::MemorySpace sp)
    {
        core::detail::AllocInfo info;
        info.name = nm;
        info.batch = batch;
        info.channels = ch;
        info.height = h;
        info.width = w;
        info.dtype = dt;
        info.bytes = bytes;
        info.space = sp;
        core::detail::logAllocBox(info);
    };

    m_inputSrc.allocate(static_cast<std::size_t>(B) * 3 * W * H);
    logBox("yolov8.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_inputStaging.allocate(static_cast<std::size_t>(B) * 3 * W * H);
    logBox("yolov8.input_staging", B, 3, H, W, core::DataType::UInt8,
           m_inputStaging.bytes(), core::MemorySpace::Host);

    m_resizeOut.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolov8.resize_out", B, 3, H, W, core::DataType::Float32,
           m_resizeOut.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolov8.input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(static_cast<std::size_t>(B) * m_srcRow * m_anchors * sizeof(float));
    logBox("yolov8.output_src", B, m_srcRow, 1, m_anchors, core::DataType::Float32,
           m_outputSrc.bytes(), core::MemorySpace::Device);

    m_outputTransposed.allocate(static_cast<std::size_t>(B) * m_srcRow * m_anchors * sizeof(float));
    logBox("yolov8.output_transposed", B, m_anchors, 1, m_srcRow,
           core::DataType::Float32, m_outputTransposed.bytes(),
           core::MemorySpace::Device);

    m_objectsPerImage = 1 + kernels::kObjectWidth * m_topK;
    m_objects.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolov8.objects", B, m_objectsPerImage, 1, 1, core::DataType::Float32,
           m_objects.bytes(), core::MemorySpace::Device);

    m_objectsHost.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolov8.objects_host", B, m_objectsPerImage, 1, 1,
           core::DataType::Float32, m_objectsHost.bytes(), core::MemorySpace::Host);
}

void YoloV8::init(const core::ModelConfig& cfg)
{
    TRT_LOG_DEBUG("YoloV8::init: entering");
    loadConfig(cfg);
    TRT_LOG_DEBUG("YoloV8::init: config loaded");
    m_engine = std::make_unique<core::TrtEngine>(cfg.engine);
    TRT_LOG_DEBUG("YoloV8::init: engine loaded");
    discoverEngineIo();
    TRT_LOG_DEBUG("YoloV8::init: io discovered (srcRow=" << m_srcRow
                 << ", anchors=" << m_anchors << ")");
    allocateBuffers();
    TRT_LOG_DEBUG("YoloV8::init: buffers allocated");
    m_detections.assign(static_cast<std::size_t>(m_cfg.batchSize), {});

    TRT_LOG_INFO("YoloV8: initialized (batch=" << m_cfg.batchSize
                 << ", srcRow=" << m_srcRow
                 << ", anchors=" << m_anchors << ")");
}

void YoloV8::setBatch(const core::Batch& batch)
{
    TRT_LOG_DEBUG("YoloV8::setBatch: entering, views=" << batch.views.size()
                 << ", validCount=" << batch.validCount);
    if (batch.views.empty())
    {
        throw std::runtime_error("yolov8: empty batch");
    }
    if (static_cast<int>(batch.views.size()) > m_cfg.batchSize)
    {
        throw std::runtime_error("yolov8: batch size larger than engine batch");
    }

    m_batch = static_cast<int>(batch.views.size());
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    TRT_LOG_DEBUG("YoloV8::setBatch: batch=" << m_batch
                 << ", srcSize=" << m_srcW << "x" << m_srcH
                 << ", dstSize=" << m_cfg.dstW << "x" << m_cfg.dstH);

    // 构造 letterbox 几何
    buildLetterboxAffine(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH, m_dst2src);

    // 把 batch 的连续 buffer 上传
    const std::size_t oneImage = 3 * static_cast<std::size_t>(m_srcH) * m_srcW;
    const std::size_t total = oneImage * batch.views.size();

    // batch.buffer 已经是"连续内存"（数据源保证），直接 H2D
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolov8: batch.buffer is null");
    }

    if (total > m_inputSrc.bytes())
    {
        // 尺寸变化触发扩容（少见）
        m_inputSrc.allocate(total);
        m_inputStaging.allocate(total);
    }

    cudaMemcpyAsync(m_inputSrc.data(), batch.buffer->data(), total,
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();

    TRT_LOG_DEBUG("YoloV8::setBatch: H2D done, " << total << " bytes");
}

void YoloV8::preprocess()
{
    TRT_LOG_DEBUG("YoloV8::preprocess: entering (batch=" << m_batch
                 << ", src=" << m_srcW << "x" << m_srcH
                 << ", dst=" << m_cfg.dstW << "x" << m_cfg.dstH << ")");
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
                                 m_normScale,
                                 m_normMean, m_normStd);
    TRT_LOG_DEBUG("YoloV8::preprocess: done");
}

void YoloV8::infer()
{
    TRT_LOG_DEBUG("YoloV8::infer: entering (batch=" << m_batch << ")");
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()))
    {
        throw std::runtime_error("yolov8: setTensorAddress failed");
    }
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolov8: enqueueV3 failed");
    }
    TRT_LOG_DEBUG("YoloV8::infer: enqueueV3 done");
}

void YoloV8::postprocess()
{
    TRT_LOG_DEBUG("YoloV8::postprocess: entering (batch=" << m_batch << ")");
    // 确保 m_detections 有 m_batch 个元素
    // （commitResult 会 move 走 m_detections，下一轮需要重新分配）
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

    kernels::transposeAnchors(m_stream.get(), m_batch,
                              m_outputSrc.asFloat(), m_srcRow, m_anchors,
                              m_outputTransposed.asFloat());

    kernels::decodeYoloV8Head(m_stream.get(), p, m_outputTransposed.asFloat(),
                              m_anchors, m_objects.asFloat());

    kernels::nmsFast(m_stream.get(), p, m_objects.asFloat(), kernels::kObjectWidth);

    // D2H + 同步
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
                continue;   // NMS 淘汰
            }
            Detection d;
            // 网络输入坐标 -> 原图坐标
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
    // 加"统计日志"
    for (int b = 0; b < m_batch; ++b)
    {
        TRT_LOG_DEBUG("YoloV8::postprocess: batch[" << b << "] detections="
                     << m_detections[static_cast<std::size_t>(b)].size());
    }
    TRT_LOG_DEBUG("YoloV8::postprocess: done");
}

void YoloV8::reset()
{
    TRT_LOG_DEBUG("YoloV8::reset: clearing " << m_detections.size()
                 << " image result(s)");
    for (auto& v : m_detections)
    {
        v.clear();
    }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("yolov8", trt_alpha::det::YoloV8);