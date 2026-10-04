// =============================================================================
//  trt_alpha :: det :: YoloV4（实现）
// -----------------------------------------------------------------------------
//  输出 [B, anchors, 1, 4+nc]（4 维，无 objectness，归一化坐标）。
// =============================================================================
#include "yolov4.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace trt_alpha::det {
namespace {

//! letterbox 仿射矩阵（居中 padding，和 YOLOv5 一致）。
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

const std::string& YoloV4::name() const noexcept
{
    static const std::string kName = "yolov4";
    return kName;
}

void YoloV4::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    // YOLOv4 官方默认值
    if (cfg.getString("conf_thresh", "").empty()) m_confThreshold = 0.4f;
    if (cfg.getString("iou_thresh",  "").empty()) m_iouThreshold  = 0.6f;

    if (m_numClass <= 0) {
        throw std::runtime_error("yolov4: num_class must be > 0");
    }

    TRT_LOG_INFO("YoloV4: config num_class=" << m_numClass
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK);
}

void YoloV4::discoverEngineIo()
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
        throw std::runtime_error("yolov4: engine must have >=1 input and >=1 output");
    }
    m_inputName = input->name;
    m_outputName = output->name;

    m_engine->setInputShape(m_inputName, nvinfer1::Dims4(m_cfg.batchSize, 3,
                                                          m_cfg.dstH, m_cfg.dstW));

    // YOLOv4 输出是 4 维 [B, anchors, 1, 4+nc]
    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 4)
    {
        throw std::runtime_error("yolov4: expect output as [batch, anchors, 1, 4+nc]");
    }
    m_anchors = static_cast<int>(outDims.d[1]);
    m_srcRow  = static_cast<int>(outDims.d[3]);   // 4 + nc
    if (m_srcRow != 4 + m_numClass)
    {
        throw std::runtime_error(
            "yolov4: output channel = " + std::to_string(m_srcRow) +
            " but 4+num_class = " + std::to_string(4 + m_numClass) +
            " (check 'num_class' in INI)");
    }
    if (outDims.d[0] > 0 && outDims.d[0] < m_cfg.batchSize)
    {
        throw std::runtime_error("yolov4: engine max batch < config batch_size");
    }
}

void YoloV4::allocateBuffers()
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
    logBox("yolov4.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_resizeOut.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolov4.resize_out", B, 3, H, W, core::DataType::Float32,
           m_resizeOut.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolov4.input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(static_cast<std::size_t>(B) * m_anchors * m_srcRow * sizeof(float));
    logBox("yolov4.output_src", B, m_srcRow, 1, m_anchors, core::DataType::Float32,
           m_outputSrc.bytes(), core::MemorySpace::Device);

    m_objectsPerImage = 1 + kernels::kObjectWidth * m_topK;
    m_objects.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolov4.objects", B, m_objectsPerImage, 1, 1, core::DataType::Float32,
           m_objects.bytes(), core::MemorySpace::Device);

    m_objectsHost.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolov4.objects_host", B, m_objectsPerImage, 1, 1,
           core::DataType::Float32, m_objectsHost.bytes(), core::MemorySpace::Host);

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()))
    {
        throw std::runtime_error("yolov4: setTensorAddress failed");
    }
}

void YoloV4::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();
    m_detections.assign(static_cast<std::size_t>(m_cfg.batchSize), {});
    TRT_LOG_INFO("YoloV4: initialized (batch=" << m_cfg.batchSize
                 << ", anchors=" << m_anchors
                 << ", srcRow=" << m_srcRow << ")");
}

void YoloV4::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yolov4: empty batch");
    }
    m_batch = static_cast<int>(batch.views.size());
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    buildLetterboxAffine(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH, m_dst2src);

    const std::size_t oneImage = 3 * static_cast<std::size_t>(m_srcH) * m_srcW;
    const std::size_t total = oneImage * batch.views.size();
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolov4: batch.buffer is null");
    }
    if (total > m_inputSrc.bytes())
    {
        m_inputSrc.allocate(total);
    }
    cudaMemcpyAsync(m_inputSrc.data(), batch.buffer->data(), total,
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();
}

void YoloV4::preprocess()
{
    kernels::resizeLetterbox(m_stream.get(), m_batch,
                             static_cast<const std::uint8_t*>(m_inputSrc.data()),
                             m_srcW, m_srcH,
                             m_resizeOut.asFloat(),
                             m_cfg.dstW, m_cfg.dstH,
                             m_padValue, m_dst2src);

    // YOLOv4: BGR->RGB + /255（和 YOLOv5 一致，swapRB=true）
    kernels::bgrToNchwNormalized(m_stream.get(), m_batch,
                                 m_resizeOut.asFloat(),
                                 m_inputNchw.asFloat(),
                                 m_cfg.dstW, m_cfg.dstH,
                                 m_normScale, m_normMean, m_normStd,
                                 /*swapRB=*/true);
}

void YoloV4::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolov4: enqueueV3 failed");
    }
}

// void YoloV4::postprocess()
// {
//     m_detections.assign(static_cast<std::size_t>(m_batch), {});

//     kernels::YoloDecodeParams p;
//     p.batch = m_batch;
//     p.numClasses = m_numClass;
//     p.topK = m_topK;
//     p.confThreshold = m_confThreshold;
//     p.iouThreshold = m_iouThreshold;

//     cudaMemsetAsync(m_objects.data(), 0,
//                     static_cast<std::size_t>(m_objectsPerImage) * m_cfg.batchSize * sizeof(float),
//                     m_stream.get());

//     // YOLOv4 专用 decode（4 维输入 + 无 objectness + 归一化坐标）
//     kernels::decodeYoloV4Head(m_stream.get(), p, m_outputSrc.asFloat(),
//                               m_anchors, m_cfg.dstW, m_cfg.dstH,
//                               m_objects.asFloat());

//     kernels::nmsFast(m_stream.get(), p, m_objects.asFloat(), kernels::kObjectWidth);

//     cudaMemcpyAsync(m_objectsHost.data(), m_objects.data(),
//                     static_cast<std::size_t>(m_objectsPerImage) * m_batch * sizeof(float),
//                     cudaMemcpyDeviceToHost, m_stream.get());
//     m_stream.synchronize();

//     const float* host = m_objectsHost.asFloat();
//     for (int b = 0; b < m_batch; ++b)
//     {
//         const float* row = host + static_cast<std::size_t>(b) * m_objectsPerImage;
//         const int count = std::clamp(static_cast<int>(row[0]), 0, m_topK);
//         m_detections[static_cast<std::size_t>(b)].clear();
//         for (int i = 0; i < count; ++i)
//         {
//             const float* o = row + 1 + i * kernels::kObjectWidth;
//             if (o[6] < 0.5f)
//             {
//                 continue;
//             }
//             Detection d;
//             const float nx = o[0], ny = o[1], nr = o[2], nb = o[3];
//             d.left   = m_dst2src.v0 * nx + m_dst2src.v1 * ny + m_dst2src.v2;
//             d.top    = m_dst2src.v3 * nx + m_dst2src.v4 * ny + m_dst2src.v5;
//             d.right  = m_dst2src.v0 * nr + m_dst2src.v1 * nb + m_dst2src.v2;
//             d.bottom = m_dst2src.v3 * nr + m_dst2src.v4 * nb + m_dst2src.v5;
//             d.confidence = o[4];
//             d.label = static_cast<int>(o[5]);
//             m_detections[static_cast<std::size_t>(b)].push_back(d);
//         }
//     }
// }

void YoloV4::postprocess()
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

    // YOLOv4 专用 decode（4 维输入 + 无 objectness + 归一化坐标）
    kernels::decodeYoloV4Head(m_stream.get(), p, m_outputSrc.asFloat(),
                              m_anchors, m_cfg.dstW, m_cfg.dstH,
                              m_objects.asFloat());

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
                continue;   // NMS 淘汰
            }
            Detection d;

            // YOLOv4: o[0..3] = (left, top, right, bottom)，归一化 0~1。
            // 1) 归一化 → 网络输入像素
            const float nx = o[0] * static_cast<float>(m_cfg.dstW);
            const float ny = o[1] * static_cast<float>(m_cfg.dstH);
            const float nw = o[2] * static_cast<float>(m_cfg.dstW);
            const float nh = o[3] * static_cast<float>(m_cfg.dstH);

            // 2) 网络输入像素 → 源图像素（m_dst2src 仿射）
            d.left   = m_dst2src.v0 * nx + m_dst2src.v1 * ny + m_dst2src.v2;
            d.top    = m_dst2src.v3 * nx + m_dst2src.v4 * ny + m_dst2src.v5;
            d.right  = m_dst2src.v0 * nw + m_dst2src.v1 * nh + m_dst2src.v2;
            d.bottom = m_dst2src.v3 * nw + m_dst2src.v4 * nh + m_dst2src.v5;

            d.confidence = o[4];
            d.label = static_cast<int>(o[5]);
            m_detections[static_cast<std::size_t>(b)].push_back(d);
        }
    }
}

void YoloV4::reset()
{
    for (auto& v : m_detections)
    {
        v.clear();
    }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("yolov4", trt_alpha::det::YoloV4);