// =============================================================================
//  trt_alpha :: depth :: YoloV26Depth（实现）
// -----------------------------------------------------------------------------
//  管线：
//      setBatch   : 原图连续内存 H2D
//      preprocess : CvPreprocessVecPlan —— letterbox 640x640 + BGR2RGB +
//                   HWC2CHW + /255，五合一单 kernel（与检测逐位相同）
//      infer      : enqueueV3
//      postprocess: CvScaleMaskPlan<float>(Bilinear) —— 裁灰边 + 双线性回原图，
//                   单 kernel；然后 D2H 成 DepthMap
//
//  为什么这里没有 interpolate：
//      导出的 yolo26n-depth.onnx 的 output0 已经是 [B,1,640,640] 的米制深度
//      （exp(clamp(-4,5)) + 上采样都在图里）。踩过这个坑：参考脚本
//      sry_yolo26_depth.py 里的 F.interpolate 是因为它直接跑 .pt 模型；ONNX 图
//      已经把这一步烘焙掉了，再插一次就是【多算一次】。用 onnxruntime 实测
//      （D:\Data\trt_exp\probe_onnx_io.py）确认过形状和值域。
// =============================================================================
#include "yolov26_depth.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"   // TRT_ALPHA_REGISTER_MODEL

#include <cstring>
#include <stdexcept>
#include <string>

namespace trt_alpha::depth {

const std::string& YoloV26Depth::name() const noexcept
{
    static const std::string kName = "yolov26_depth";
    return kName;
}

void YoloV26Depth::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    TRT_LOG_INFO("YoloV26Depth: config scale=" << m_normScale
                 << " pad_value=" << m_padValue);
}

void YoloV26Depth::discoverEngineIo()
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
        throw std::runtime_error("yolov26_depth: engine must have >=1 input and >=1 output");
    }
    m_inputName  = input->name;
    m_outputName = output->name;

    // 输入形状以引擎声明为真相源
    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims == 4)
    {
        // [B, 1, H, W] —— 期望的形状
        if (outDims.d[1] != 1)
        {
            throw std::runtime_error(
                "yolov26_depth: output0 channel = " + std::to_string(outDims.d[1]) +
                " but a depth map must have exactly 1 channel");
        }
        m_outH = static_cast<int>(outDims.d[2]);
        m_outW = static_cast<int>(outDims.d[3]);
    }
    else if (outDims.nbDims == 3)
    {
        // [B, H, W] —— 官方 DepthPredictor 同款兜底
        m_outH = static_cast<int>(outDims.d[1]);
        m_outW = static_cast<int>(outDims.d[2]);
    }
    else
    {
        throw std::runtime_error("yolov26_depth: expect output0 as [B,1,H,W] or [B,H,W]");
    }

    if (outDims.d[0] > 0 && outDims.d[0] != m_cfg.batchSize)
    {
        throw std::runtime_error(
            "yolov26_depth: output0 batch = " + std::to_string(outDims.d[0]) +
            " but input batch = " + std::to_string(m_cfg.batchSize) +
            " (engine profile is the single source of truth; check 'batch_size')");
    }
    if (m_outH <= 0 || m_outW <= 0)
    {
        throw std::runtime_error("yolov26_depth: output0 spatial dims must be > 0");
    }
}

void YoloV26Depth::allocateBuffers()
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
    logBox("yolov26_depth.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * 3 * W * H * sizeof(float));
    logBox("yolov26_depth.input_nchw", B, 3, W, H, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_depthNet.allocate(static_cast<std::size_t>(B) * m_outH * m_outW * sizeof(float));
    logBox("yolov26_depth.depth_net", B, 1, m_outH, m_outW, core::DataType::Float32,
           m_depthNet.bytes(), core::MemorySpace::Device);

    // scale_mask 的目标尺寸此时还不知道（要等 setBatch 拿到原图），先按网络尺寸
    // 预留；setBatch 里按真实 srcW*srcH 扩容。
    const std::size_t guess = static_cast<std::size_t>(B) * m_outH * m_outW * sizeof(float);
    m_depthOrig.allocate(guess);
    m_depthHost.allocate(guess);

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_depthNet.data()))
    {
        throw std::runtime_error("yolov26_depth: setTensorAddress failed");
    }
}

void YoloV26Depth::init(const core::ModelConfig& cfg)
{
    TRT_LOG_DEBUG("YoloV26Depth::init: entering");
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();

    TRT_LOG_INFO("YoloV26Depth: initialized (batch=" << m_cfg.batchSize
                 << ", net=" << m_cfg.dstW << "x" << m_cfg.dstH
                 << ", depth=" << m_outW << "x" << m_outH << ")");
}

void YoloV26Depth::ensurePlans()
{
    const bool geomChanged = !m_pre ||
        !m_pre->matches(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH);

    if (!geomChanged)
    {
        return;
    }

    m_pre = std::make_unique<kernels::ops::CvPreprocessVecPlan>(
        kernels::ops::CvPreprocessVecPlan::byDivisor(
            m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH,
            static_cast<double>(m_padValue), m_normScale));
    if (!m_pre->ok())
    {
        throw std::runtime_error("yolov26_depth: preprocess plan construction failed");
    }

    // 反变换：从网络空间裁掉灰边，双线性缩回原图。srcW/srcH 是【输出】尺寸，
    // 裁剪窗由 computeLetterBox(srcW,srcH,netW,netH) 推出 —— 跟上面的 letterbox
    // 是同一个函数、同一组参数。
    m_sm = std::make_unique<kernels::ops::CvScaleMaskPlan<float>>(
        m_srcW, m_srcH, m_outW, m_outH, kernels::ops::MaskInterp::Bilinear);
    if (!m_sm->ok())
    {
        throw std::runtime_error("yolov26_depth: scale_mask plan construction failed");
    }

    const kernels::ops::LetterBoxInfo& lb = m_pre->info();
    TRT_LOG_DEBUG("YoloV26Depth: plans rebuilt  " << m_srcW << "x" << m_srcH
                 << " -> new " << lb.newW << "x" << lb.newH
                 << " crop=[" << (lb.padTop) << ":" << (lb.padTop + lb.newH)
                 << "," << lb.padLeft << ":" << (lb.padLeft + lb.newW) << "]");

    // 目标尺寸变化 -> 设备/主机缓冲跟着扩
    const std::size_t need =
        static_cast<std::size_t>(m_cfg.batchSize) * m_sm->outElems() * sizeof(float);
    if (need > m_depthOrig.bytes())
    {
        m_depthOrig.allocate(need);
        m_depthHost.allocate(need);
    }
}

void YoloV26Depth::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yolov26_depth: empty batch");
    }

    m_batch = requireBatchCapacity(*this, batch, "yolov26_depth");
    m_srcH  = batch.views[0].height;
    m_srcW  = batch.views[0].width;
    if (m_srcW <= 0 || m_srcH <= 0)
    {
        throw std::runtime_error("yolov26_depth: invalid source size " +
                                 std::to_string(m_srcW) + "x" + std::to_string(m_srcH));
    }

    ensurePlans();

    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolov26_depth: batch.buffer is null");
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

    TRT_LOG_DEBUG("YoloV26Depth::setBatch: H2D done, " << total
                 << " bytes (batch=" << m_batch << ")");
}

void YoloV26Depth::preprocess()
{
    if (!m_pre)
    {
        throw std::runtime_error("yolov26_depth: preprocess called before setBatch");
    }
    m_pre->launch(m_stream.get(), m_batch,
                  static_cast<const unsigned char*>(m_inputSrc.data()),
                  m_inputNchw.asFloat());
    if (!m_pre->ok())
    {
        throw std::runtime_error("yolov26_depth: preprocess kernel launch failed");
    }
}

void YoloV26Depth::infer()
{
    if (!m_engine->context()->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolov26_depth: enqueueV3 failed");
    }
}

void YoloV26Depth::postprocess()
{
    m_depths.assign(static_cast<std::size_t>(m_batch), {});

    m_sm->launch(m_stream.get(), m_batch, m_depthNet.asFloat(),
                 m_depthOrig.asFloat(), /*channels=*/1);
    if (!m_sm->ok())
    {
        throw std::runtime_error("yolov26_depth: scale_mask kernel launch failed");
    }

    const std::size_t elems = static_cast<std::size_t>(m_batch) * m_sm->outElems();
    cudaMemcpyAsync(m_depthHost.data(), m_depthOrig.data(),
                    elems * sizeof(float), cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    const float* src      = m_depthHost.asFloat();
    const std::size_t per = m_sm->outElems();

    for (int b = 0; b < m_batch; ++b)
    {
        DepthMap& dm = m_depths[static_cast<std::size_t>(b)];
        dm.width  = m_sm->outW();
        dm.height = m_sm->outH();
        dm.data.assign(src + static_cast<std::size_t>(b) * per,
                       src + static_cast<std::size_t>(b + 1) * per);
    }

    TRT_LOG_DEBUG("YoloV26Depth::postprocess: " << m_batch << " depth map(s) "
                 << m_sm->outW() << "x" << m_sm->outH());
}

void YoloV26Depth::reset()
{
    for (auto& d : m_depths)
    {
        d.data.clear();
    }
    m_batch = 0;
}

}  // namespace trt_alpha::depth

TRT_ALPHA_REGISTER_MODEL("yolov26_depth", trt_alpha::depth::YoloV26Depth);
