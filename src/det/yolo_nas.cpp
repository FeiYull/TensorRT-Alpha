// =============================================================================
//  trt_alpha :: det :: YoloNas（实现）
// =============================================================================
#include "yolo_nas.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace trt_alpha::det {
namespace {

//! letterbox 仿射（按 636×636 算，不是 640×640）。
void buildNasLetterboxAffine(int srcW, int srcH, int dstW, int dstH,
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

const std::string& YoloNas::name() const noexcept
{
    static const std::string kName = "yolo_nas";
    return kName;
}

void YoloNas::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    // YOLO-NAS 官方默认值
    if (cfg.getString("iou_thresh", "").empty()) m_iouThreshold = 0.7f;

    if (m_numClass <= 0) {
        throw std::runtime_error("yolo_nas: num_class must be > 0");
    }

    // YOLO-NAS 特有字段
    // 注意：pad 依赖 m_cfg.dstW/dstH，而这两个值是引擎真相 —— 要等
    // discoverEngineIo() 里的 applyInputShape() 按引擎声明写回后才有效。
    // 此处（loadConfig）dst 仍为 0，若在这里算 pad 会得到 (0-resize)/2 的负值，
    // 让 copyWithPaddingKernel 把 letterbox 结果静默错位（不越界、不报错）。
    // pad 的求值因此收口在 discoverEngineIo()：引擎形状落定后立即派生。
    m_resizeW = cfg.getInt("resize_w", 636);
    m_resizeH = cfg.getInt("resize_h", 636);

    TRT_LOG_INFO("YoloNas: config num_class=" << m_numClass
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK
                 << " resize=" << m_resizeW << "x" << m_resizeH);
}

void YoloNas::discoverEngineIo()
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
        throw std::runtime_error("yolo_nas: engine must have >=1 input and >=1 output");
    }
    m_inputName = input->name;
    m_outputName = output->name;

    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    // 此刻 m_cfg.dstW/dstH 已被引擎声明写回，是 letterbox 目标尺寸的唯一真相源。
    // 在这里（且只在这里）派生 pad：
    //   * resize 必须为正；
    //   * resize 不得大于 dst —— 否则 pad 为负，copyWithPaddingKernel 的判据
    //     (dstY >= padTop && dstY < srcH + padTop) 会退化成"只写左上角一块"，
    //     数学上不越界、不崩溃，于是静默给错结果。按"配置与引擎能力不符即显式失败"。
    if (m_resizeW <= 0 || m_resizeH <= 0)
    {
        throw std::runtime_error("yolo_nas: resize_w/resize_h must be > 0");
    }
    if (m_resizeW > m_cfg.dstW || m_resizeH > m_cfg.dstH)
    {
        throw std::runtime_error(
            "yolo_nas: resize " + std::to_string(m_resizeW) + "x" +
            std::to_string(m_resizeH) + " exceeds engine input " +
            std::to_string(m_cfg.dstW) + "x" + std::to_string(m_cfg.dstH) +
            " (would yield negative pad)");
    }
    m_padTop  = (m_cfg.dstH - m_resizeH) / 2;
    m_padLeft = (m_cfg.dstW - m_resizeW) / 2;
    TRT_LOG_INFO("YoloNas: resize=" << m_resizeW << "x" << m_resizeH
                 << " dst=" << m_cfg.dstW << "x" << m_cfg.dstH
                 << " pad=(" << m_padTop << "," << m_padLeft << ")");

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 3)
    {
        throw std::runtime_error("yolo_nas: expect output as [batch, anchors, 4+nc]");
    }
    m_anchors = static_cast<int>(outDims.d[1]);
    m_srcRow  = static_cast<int>(outDims.d[2]);
    if (m_srcRow != 4 + m_numClass)
    {
        throw std::runtime_error(
            "yolo_nas: output channel = " + std::to_string(m_srcRow) +
            " but 4+num_class = " + std::to_string(4 + m_numClass));
    }
}

void YoloNas::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;
    const std::size_t dstArea = static_cast<std::size_t>(H) * W;
    const std::size_t oneImageF32 = 3 * dstArea * sizeof(float);
    const std::size_t resizeArea = static_cast<std::size_t>(m_resizeH) * m_resizeW;
    const std::size_t oneResizeF32 = 3 * resizeArea * sizeof(float);

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
    logBox("yolo_nas.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_resizeOut.allocate(static_cast<std::size_t>(B) * oneResizeF32);
    logBox("yolo_nas.resize_out", B, 3, m_resizeH, m_resizeW,
           core::DataType::Float32, m_resizeOut.bytes(), core::MemorySpace::Device);

    m_inputPad.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolo_nas.input_pad", B, 3, H, W, core::DataType::Float32,
           m_inputPad.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * oneImageF32);
    logBox("yolo_nas.input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(static_cast<std::size_t>(B) * m_anchors * m_srcRow * sizeof(float));
    logBox("yolo_nas.output_src", B, m_srcRow, 1, m_anchors,
           core::DataType::Float32, m_outputSrc.bytes(), core::MemorySpace::Device);

    m_objectsPerImage = 1 + kernels::kObjectWidth * m_topK;
    m_objects.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolo_nas.objects", B, m_objectsPerImage, 1, 1,
           core::DataType::Float32, m_objects.bytes(), core::MemorySpace::Device);

    m_objectsHost.allocate(static_cast<std::size_t>(B) * m_objectsPerImage * sizeof(float));
    logBox("yolo_nas.objects_host", B, m_objectsPerImage, 1, 1,
           core::DataType::Float32, m_objectsHost.bytes(), core::MemorySpace::Host);

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()))
    {
        throw std::runtime_error("yolo_nas: setTensorAddress failed");
    }
}

void YoloNas::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();
    m_detections.assign(static_cast<std::size_t>(m_cfg.batchSize), {});
    TRT_LOG_INFO("YoloNas: initialized (batch=" << m_cfg.batchSize
                 << ", anchors=" << m_anchors
                 << ", srcRow=" << m_srcRow << ")");
}

void YoloNas::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yolo_nas: empty batch");
    }
    m_batch = requireBatchCapacity(*this, batch, "yolo_nas");
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    // m_dst2src 按 636×636 算（不是 640×640）
    buildNasLetterboxAffine(m_srcW, m_srcH, m_resizeW, m_resizeH, m_dst2src);

    const std::size_t oneImage = 3 * static_cast<std::size_t>(m_srcH) * m_srcW;
    const std::size_t total = oneImage * batch.views.size();
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolo_nas: batch.buffer is null");
    }
    if (total > m_inputSrc.bytes())
    {
        m_inputSrc.allocate(total);
    }
    cudaMemcpyAsync(m_inputSrc.data(), batch.buffer->data(), total,
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();
}

void YoloNas::preprocess()
{
    // 第 1 步：letterbox 到 636×636
    kernels::resizeLetterbox(m_stream.get(), m_batch,
                             static_cast<const std::uint8_t*>(m_inputSrc.data()),
                             m_srcW, m_srcH,
                             m_resizeOut.asFloat(),
                             m_resizeW, m_resizeH,
                             m_padValue, m_dst2src);

    // 第 2 步：把 636×636 放到 640×640 的 (padTop, padLeft) 位置
    kernels::copyWithPadding(m_stream.get(), m_batch,
                             m_resizeOut.asFloat(), m_resizeW, m_resizeH,
                             m_inputPad.asFloat(), m_cfg.dstW, m_cfg.dstH,
                             m_padValue, m_padTop, m_padLeft);

    // 第 3 步：BGR->RGB + /255 + NCHW
    kernels::bgrToNchwNormalized(m_stream.get(), m_batch,
                                 m_inputPad.asFloat(),
                                 m_inputNchw.asFloat(),
                                 m_cfg.dstW, m_cfg.dstH,
                                 m_normScale, m_normMean, m_normStd,
                                 /*swapRB=*/true);
}

void YoloNas::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolo_nas: enqueueV3 failed");
    }
}

void YoloNas::postprocess()
{
    m_detections.assign(static_cast<std::size_t>(m_batch), {});

    kernels::YoloDecodeParams p;
    p.batch = m_batch;
    p.numClasses = m_numClass;
    p.topK = m_topK;
    p.confThreshold = m_confThreshold;
    p.iouThreshold = m_iouThreshold;

    // 清零范围与 D2H 拷贝范围同源（都用 m_batch），不做两套口径。
    cudaMemsetAsync(m_objects.data(), 0,
                    static_cast<std::size_t>(m_objectsPerImage) * m_batch * sizeof(float),
                    m_stream.get());

    kernels::decodeYoloNasHead(m_stream.get(), p, m_outputSrc.asFloat(),
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

            // YOLO-NAS: o[0..3] 是 640×640 上的像素 xyxy。
            // 1) 减 pad，映射回 636×636
            float x0 = o[0] - static_cast<float>(m_padLeft);
            float y0 = o[1] - static_cast<float>(m_padTop);
            float x1 = o[2] - static_cast<float>(m_padLeft);
            float y1 = o[3] - static_cast<float>(m_padTop);

            // 2) m_dst2src 仿射（按 636×636 算）映射回源图
            d.left   = m_dst2src.v0 * x0 + m_dst2src.v1 * y0 + m_dst2src.v2;
            d.top    = m_dst2src.v3 * x0 + m_dst2src.v4 * y0 + m_dst2src.v5;
            d.right  = m_dst2src.v0 * x1 + m_dst2src.v1 * y1 + m_dst2src.v2;
            d.bottom = m_dst2src.v3 * x1 + m_dst2src.v4 * y1 + m_dst2src.v5;

            d.confidence = o[4];
            d.label = static_cast<int>(o[5]);
            m_detections[static_cast<std::size_t>(b)].push_back(d);
        }
    }
}

void YoloNas::reset()
{
    for (auto& v : m_detections)
    {
        v.clear();
    }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("yolo_nas", trt_alpha::det::YoloNas);