// =============================================================================
//  trt_alpha :: det :: YoloV8Pose（实现）
// =============================================================================
#include "yolov8_pose.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace trt_alpha::seg {
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

const std::string& YoloV8Pose::name() const noexcept
{
    static const std::string kName = "yolov8_pose";
    return kName;
}

void YoloV8Pose::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    // pose 官方默认 iou
    if (cfg.getString("iou_thresh", "").empty()) m_iouThreshold = 0.7f;

    // pose 特有
    m_numKpts = cfg.getInt("num_kpts", 17);

    TRT_LOG_INFO("YoloV8Pose: config num_kpts=" << m_numKpts
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK);
}

void YoloV8Pose::discoverEngineIo()
{
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput)   { m_inputName = t.name;  continue; }
        if (!t.isInput)  { m_outputName = t.name; continue; }
    }
    if (m_inputName.empty() || m_outputName.empty())
    {
        throw std::runtime_error("yolov8_pose: missing input/output tensor");
    }

    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 3)
    {
        throw std::runtime_error("yolov8_pose: expect output0 as [B, 5+3*K, anchors]");
    }
    m_srcRow  = static_cast<int>(outDims.d[1]);
    m_anchors = static_cast<int>(outDims.d[2]);
    if (m_srcRow != 5 + m_numKpts * 3)
    {
        throw std::runtime_error(
            "yolov8_pose: output0 channel = " + std::to_string(m_srcRow) +
            " but 5 + 3*num_kpts = " + std::to_string(5 + m_numKpts * 3));
    }
}

void YoloV8Pose::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;
    const std::size_t dstArea = std::size_t(H) * W;
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

    m_inputSrc.allocate(std::size_t(B) * 3 * W * H);
    logBox("yolov8_pose.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);
    m_resizeOut.allocate(std::size_t(B) * oneImageF32);
    logBox("yolov8_pose.resize_out", B, 3, H, W, core::DataType::Float32,
           m_resizeOut.bytes(), core::MemorySpace::Device);
    m_inputNchw.allocate(std::size_t(B) * oneImageF32);
    logBox("yolov8_pose.input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(std::size_t(B) * m_srcRow * m_anchors * sizeof(float));
    logBox("yolov8_pose.output0", B, m_srcRow, 1, m_anchors,
           core::DataType::Float32, m_outputSrc.bytes(), core::MemorySpace::Device);

    m_outputTransposed.allocate(std::size_t(B) * m_srcRow * m_anchors * sizeof(float));
    logBox("yolov8_pose.output0_transposed", B, m_anchors, 1, m_srcRow,
           core::DataType::Float32, m_outputTransposed.bytes(),
           core::MemorySpace::Device);

    m_objectsRow = 7 + m_numKpts * 3;
    m_objectsPerImage = 1 + m_objectsRow * m_topK;
    m_objects.allocate(std::size_t(B) * m_objectsPerImage * sizeof(float));
    logBox("yolov8_pose.objects", B, m_objectsPerImage, 1, 1,
           core::DataType::Float32, m_objects.bytes(), core::MemorySpace::Device);
    m_objectsHost.allocate(std::size_t(B) * m_objectsPerImage * sizeof(float));

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()))
    {
        throw std::runtime_error("yolov8_pose: setTensorAddress failed");
    }
}

void YoloV8Pose::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();
    m_keypoints.assign(static_cast<std::size_t>(m_cfg.batchSize), {});
    TRT_LOG_INFO("YoloV8Pose: initialized (batch=" << m_cfg.batchSize
                 << ", anchors=" << m_anchors
                 << ", srcRow=" << m_srcRow << ")");
}

void YoloV8Pose::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yolov8_pose: empty batch");
    }
    m_batch = static_cast<int>(batch.views.size());
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    buildLetterboxAffine(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH, m_dst2src);

    const std::size_t oneImage = 3 * std::size_t(m_srcH) * m_srcW;
    const std::size_t total = oneImage * batch.views.size();
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolov8_pose: batch.buffer is null");
    }
    if (total > m_inputSrc.bytes())
    {
        m_inputSrc.allocate(total);
    }
    cudaMemcpyAsync(m_inputSrc.data(), batch.buffer->data(), total,
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();
}

void YoloV8Pose::preprocess()
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
                                 m_normScale, m_normMean, m_normStd,
                                 /*swapRB=*/true);
}

void YoloV8Pose::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolov8_pose: enqueueV3 failed");
    }
}

void YoloV8Pose::postprocess()
{
    m_keypoints.assign(static_cast<std::size_t>(m_batch), {});

    kernels::YoloDecodeParams p;
    p.batch = m_batch;
    p.topK = m_topK;
    p.confThreshold = m_confThreshold;
    p.iouThreshold = m_iouThreshold;

    cudaMemsetAsync(m_objects.data(), 0,
                    std::size_t(m_objectsPerImage) * m_cfg.batchSize * sizeof(float),
                    m_stream.get());

    // 1. transpose output0: [B, 56, 8400] -> [B, 8400, 56]
    kernels::transposeAnchors(m_stream.get(), m_batch,
                              m_outputSrc.asFloat(), m_srcRow, m_anchors,
                              m_outputTransposed.asFloat());

    // 2. decode（带关键点）
    kernels::decodeYoloV8PoseHead(m_stream.get(), p,
                                  m_outputTransposed.asFloat(), m_anchors,
                                  m_numKpts, m_objects.asFloat());

    // 3. NMS（只读前 7 个字段）
    kernels::nmsFast(m_stream.get(), p, m_objects.asFloat(), m_objectsRow);

    // 4. D2H
    cudaMemcpyAsync(m_objectsHost.data(), m_objects.data(),
                    std::size_t(m_objectsPerImage) * m_batch * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    // 5. CPU 后处理：m_dst2src 变换关键点
    const float* host = m_objectsHost.asFloat();
    for (int b = 0; b < m_batch; ++b)
    {
        const float* row = host + std::size_t(b) * m_objectsPerImage;
        const int count = std::clamp(static_cast<int>(row[0]), 0, m_topK);
        for (int i = 0; i < count; ++i)
        {
            const float* o = row + 1 + i * m_objectsRow;
            if (o[6] < 0.5f) { continue; }

            kpt::KeypointResult kr;
            // 框：网络输入坐标 -> 原图
            const float x0 = o[0], y0 = o[1], x1 = o[2], y1 = o[3];
            kr.box.left   = m_dst2src.v0 * x0 + m_dst2src.v1 * y0 + m_dst2src.v2;
            kr.box.top    = m_dst2src.v3 * x0 + m_dst2src.v4 * y0 + m_dst2src.v5;
            kr.box.right  = m_dst2src.v0 * x1 + m_dst2src.v1 * y1 + m_dst2src.v2;
            kr.box.bottom = m_dst2src.v3 * x1 + m_dst2src.v4 * y1 + m_dst2src.v5;
            kr.box.confidence = o[4];
            kr.box.label = static_cast<int>(o[5]);

            // 17 个关键点：网络输入坐标 -> 原图
            kr.keypoints.reserve(m_numKpts);
            for (int k = 0; k < m_numKpts; ++k)
            {
                const float kx = o[7 + k * 3 + 0];
                const float ky = o[7 + k * 3 + 1];
                const float kc = o[7 + k * 3 + 2];
                kpt::Keypoint kp;
                kp.x = m_dst2src.v0 * kx + m_dst2src.v1 * ky + m_dst2src.v2;
                kp.y = m_dst2src.v3 * kx + m_dst2src.v4 * ky + m_dst2src.v5;
                kp.confidence = kc;
                kr.keypoints.push_back(kp);
            }
            m_keypoints[std::size_t(b)].push_back(std::move(kr));
        }
    }
}

void YoloV8Pose::reset()
{
    for (auto& v : m_keypoints) { v.clear(); }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("yolov8_pose", trt_alpha::seg::YoloV8Pose);