// =============================================================================
//  trt_alpha :: seg :: U2Net（实现）
// =============================================================================
#include "u2net.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include "trt_alpha/kernels/legacy/u2net_reduce.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_alpha::seg {
namespace {

//! 计算 src->dst 的缩放仿射（u2net 是"非等比缩放"，x 和 y 独立）。
void buildU2NetAffine(int srcW, int srcH, int dstW, int dstH,
                      kernels::AffineMat& src2dst,
                      kernels::AffineMat& dst2src)
{
    const float scaleX = static_cast<float>(dstW) / static_cast<float>(srcW);
    const float scaleY = static_cast<float>(dstH) / static_cast<float>(srcH);
    src2dst.v0 = scaleX;  src2dst.v1 = 0.f;  src2dst.v2 = 0.f;
    src2dst.v3 = 0.f;     src2dst.v4 = scaleY; src2dst.v5 = 0.f;

    const float invX = 1.f / scaleX;
    const float invY = 1.f / scaleY;
    dst2src.v0 = invX;   dst2src.v1 = 0.f;   dst2src.v2 = 0.f;
    dst2src.v3 = 0.f;    dst2src.v4 = invY;  dst2src.v5 = 0.f;
}

}  // namespace

const std::string& U2Net::name() const noexcept
{
    static const std::string kName = "u2net";
    return kName;
}

void U2Net::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    // u2net 特有
    m_postScale = cfg.getFloat("post_scale", 255.f);

    TRT_LOG_INFO("U2Net: config num_class=" << m_numClass
                 << " norm_scale=" << m_normScale
                 << " post_scale=" << m_postScale
                 << " mean=(" << m_normMean[0] << "," << m_normMean[1] << "," << m_normMean[2] << ")"
                 << " std=(" << m_normStd[0] << "," << m_normStd[1] << "," << m_normStd[2] << ")");
}

void U2Net::discoverEngineIo()
{
    // 口径：第一个输入 + 第一个非输入输出（与其他模型一致；多输出引擎请显式指定）
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput)  { if (m_inputName.empty())  { m_inputName  = t.name; } }
        else            { if (m_outputName.empty()) { m_outputName = t.name; } }
    }
    if (m_inputName.empty() || m_outputName.empty())
    {
        throw std::runtime_error("u2net: missing input/output tensor");
    }

    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 4 || outDims.d[1] != 1)
    {
        throw std::runtime_error("u2net: expect output as [B, 1, H, W]");
    }
    TRT_LOG_INFO("U2Net: io names: input='" << m_inputName
                 << "' output='" << m_outputName << "'");
}

void U2Net::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int DH = m_cfg.dstH;
    const int DW = m_cfg.dstW;

    auto logBox = [&](const char* nm, int batch, int ch, int h, int w,
                      core::DataType dt, std::size_t bytes, core::MemorySpace sp)
    {
        core::detail::AllocInfo info;
        info.name = nm; info.batch = batch; info.channels = ch;
        info.height = h; info.width = w; info.dtype = dt;
        info.bytes = bytes; info.space = sp;
        core::detail::logAllocBox(info);
    };

    // 输入 buffer（srcH/srcW 在 setBatch 里确定，先按 dst 尺寸分配）
    m_inputSrc.allocate(std::size_t(B) * 3 * DH * DW * sizeof(float));
    logBox("u2net.input_src", B, 3, DH, DW, core::DataType::Float32,
           m_inputSrc.bytes(), core::MemorySpace::Device);
    m_inputResize.allocate(std::size_t(B) * 3 * DH * DW * sizeof(float));
    logBox("u2net.input_resize", B, 3, DH, DW, core::DataType::Float32,
           m_inputResize.bytes(), core::MemorySpace::Device);
    m_inputNchw.allocate(std::size_t(B) * 3 * DH * DW * sizeof(float));
    logBox("u2net.input_nchw", B, 3, DH, DW, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_maxValDevice.allocate(std::size_t(B) * sizeof(float));
    m_minValDevice.allocate(std::size_t(B) * sizeof(float));
    m_postMaxDevice.allocate(std::size_t(B) * sizeof(float));

    m_outputSrc.allocate(std::size_t(B) * 1 * DH * DW * sizeof(float));
    logBox("u2net.output_src", B, 1, DH, DW, core::DataType::Float32,
           m_outputSrc.bytes(), core::MemorySpace::Device);

    // output_resize 依赖 srcH/srcW，setBatch 里再分配
    m_outputResize.allocate(std::size_t(B) * 1 * DH * DW * sizeof(float));
    m_outputResizeHost.allocate(std::size_t(B) * 1 * DH * DW * sizeof(float));

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()))
    {
        throw std::runtime_error("u2net: setTensorAddress failed");
    }
}

void U2Net::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();
    m_segmentations.assign(static_cast<std::size_t>(m_cfg.batchSize), {});
    TRT_LOG_INFO("U2Net: initialized (batch=" << m_cfg.batchSize
                 << ", dst=" << m_cfg.dstW << "x" << m_cfg.dstH << ")");
}

void U2Net::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("u2net: empty batch");
    }
    m_batch = requireBatchCapacity(*this, batch, "u2net");
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    buildU2NetAffine(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH, m_src2dst, m_dst2src);

    // src 尺寸的 buffer（m_inputSrc 要覆盖 srcH*srcW）
    const std::size_t oneSrc = std::size_t(m_srcH) * m_srcW * 3;
    if (m_inputSrc.bytes() < std::size_t(m_batch) * oneSrc * sizeof(float))
    {
        m_inputSrc.allocate(std::size_t(m_batch) * oneSrc * sizeof(float));
    }

    // output_resize（src 尺寸）
    const std::size_t oneSrcGray = std::size_t(m_srcH) * m_srcW;
    if (m_outputResize.bytes() < std::size_t(m_batch) * oneSrcGray * sizeof(float))
    {
        m_outputResize.allocate(std::size_t(m_batch) * oneSrcGray * sizeof(float));
    }
    if (m_outputResizeHost.bytes() < std::size_t(m_batch) * oneSrcGray * sizeof(float))
    {
        m_outputResizeHost.allocate(std::size_t(m_batch) * oneSrcGray * sizeof(float));
    }

    // batch.buffer 是 uint8：H2D uint8 -> GPU kernel -> float
    const std::size_t totalU8 = std::size_t(m_batch) * oneSrc;
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("u2net: batch.buffer is null");
    }
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

void U2Net::preprocess()
{
    // 1. BGR -> RGB（in-place，保持 HWC）
    kernels::bgrToRgbHwc(m_stream.get(), m_batch, m_inputSrc.asFloat(),
                         m_srcW, m_srcH);

    // 2. 非等比 resize：原图 -> 320×320（无 padding）
    kernels::resizeNoPadding(m_stream.get(), m_batch,
                             m_inputSrc.asFloat(),
                             m_srcW, m_srcH,
                             m_inputResize.asFloat(),
                             m_cfg.dstW, m_cfg.dstH,
                             /*isGray=*/false,
                             m_dst2src);

    // 3. divByMax：每张图除以其 RGB 最大值
    kernels::reduceMax(m_stream.get(),
                       m_inputResize.asFloat(),
                       m_maxValDevice.asFloat(),
                       m_batch,
                       3 * m_cfg.dstH * m_cfg.dstW);
    kernels::divByMax(m_stream.get(), m_batch, m_inputResize.asFloat(),
                      m_cfg.dstW, m_cfg.dstH, 3, m_maxValDevice.asFloat());

    // 4. 归一化 + HWC->CHW（swapRB=false，前面已 BGR->RGB）
    kernels::bgrToNchwNormalized(m_stream.get(), m_batch,
                                 m_inputResize.asFloat(),
                                 m_inputNchw.asFloat(),
                                 m_cfg.dstW, m_cfg.dstH,
                                 m_normScale, m_normMean, m_normStd,
                                 /*swapRB=*/false);
}

void U2Net::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("u2net: enqueueV3 failed");
    }
}

void U2Net::postprocess()
{
    m_segmentations.assign(static_cast<std::size_t>(m_batch), {});

    const int DH = m_cfg.dstH;
    const int DW = m_cfg.dstW;

    // 1. 算每张图输出的 min/max
    kernels::reduceMinMax(m_stream.get(),
                          m_outputSrc.asFloat(),
                          m_minValDevice.asFloat(),
                          m_postMaxDevice.asFloat(),
                          m_batch,
                          DH * DW);

    // 2. 归一化到 [0, postScale]
    kernels::normPred(m_stream.get(), m_batch, m_outputSrc.asFloat(),
                      DW, DH, m_postScale,
                      m_minValDevice.asFloat(), m_postMaxDevice.asFloat());

    // 3. 非等比 resize：网络输出 320×320 -> 原图（无 padding）
    kernels::resizeNoPadding(m_stream.get(), m_batch,
                             m_outputSrc.asFloat(),
                             DW, DH,
                             m_outputResize.asFloat(),
                             m_srcW, m_srcH,
                             /*isGray=*/true,
                             m_src2dst);

    // 4. D2H
    cudaMemcpyAsync(m_outputResizeHost.data(), m_outputResize.data(),
                    std::size_t(m_batch) * std::size_t(m_srcH) * m_srcW * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    // 5. 转 uint8，构造 Segmentation
    const float* host = static_cast<const float*>(m_outputResizeHost.data());
    for (int b = 0; b < m_batch; ++b)
    {
        auto maskBuf = core::Buffer::createHost(m_srcW, m_srcH, 1, core::DataType::UInt8);
        std::uint8_t* maskData = maskBuf->mutableData();
        const float* maskSrc = host + std::size_t(b) * m_srcH * m_srcW;
        for (std::size_t i = 0; i < std::size_t(m_srcH) * m_srcW; ++i)
        {
            const float v = maskSrc[i];
            maskData[i] = static_cast<std::uint8_t>(std::clamp(v, 0.f, 255.f));
        }

        seg::Segmentation s;
        // 无框（box.label 默认 -1），只填 mask
        s.mask = maskBuf->view();
        s.maskOwner = maskBuf;
        m_segmentations[std::size_t(b)].push_back(std::move(s));
    }
}

void U2Net::reset()
{
    for (auto& v : m_segmentations) { v.clear(); }
    m_batch = 0;
}

}  // namespace trt_alpha::seg

TRT_ALPHA_REGISTER_MODEL("u2net", trt_alpha::seg::U2Net);