// =============================================================================
//  trt_alpha :: seg :: YoloV26Seg（实现）
// -----------------------------------------------------------------------------
//  管线：
//      setBatch   : 原图连续内存 H2D
//      preprocess : CvPreprocessVecPlan —— letterbox 640x640 + BGR2RGB +
//                   HWC2CHW + /255（与 det 逐位相同的那个融合核）
//      infer      : enqueueV3
//      postprocess: CvProprocessFusedVecYoloV26Plan（extra=32）—— decode 已在
//                   图里，算子只做 conf 过滤 + NMS；随后**按 chunk 批量**做
//                       CvMaskCoeffDotPlan（系数 x proto -> 160x160 logits）
//                    -> CvScaleMaskPlan<float>（ATen bilinear 上采样到 640x640）
//                    -> binarizeCropMasks（>0 二值化 + 按框裁剪，**右/下严格
//                       小于**；同一趟把 up 槽位就地改写成 0/1 float）
//                    -> 一次 D2H（uint8）+ 一次 sync（整个 chunk 共用）
//                    -> 全零则丢弃（官方 keep = masks.amax() > 0，计数由核给出）
//  输出：mask      = 640x640 网络空间（对齐 py 口径 / 对拍，保留原样）
//        maskOrig  = 原图尺寸（= scale_masks 那一步），**仅 [model] mask_orig = true
//                    时生成**；默认不生成，因为内置 renderer 不用它。走的是
//                    CvScaleMaskPlan<float> + thresholdMaskToU8，即 py 的
//                    F.interpolate(bilinear) + "> 0.5"，全在 device 上。
//
//  **整条后处理没有一个 host 侧数学循环**：device 只回传成品 uint8 mask
//  （409 KB/实例），不再回传 640x640 的 float 平面（1.6 MB/实例）。
//
//  为什么逐个实例算、却按 chunk 拷贝：
//      官方顺序要求先把 logits 上采样到 640x640 再二值化，所以每个实例都要一份
//      640x640 的 float 工作区。按 top_k = 300 一次性铺开要 ~500 MB，而一张真图
//      只有十几个实例。逐个 launch 则工作区只要 kMaskChunk 份，同时把每实例一次
//      的 sync 折成每 chunk 一次 —— 实测 39 实例/批时 sync 从 39 次降到 5 次。
//
//  参考实现：ultralytics/models/yolo/segment/predict.py::construct_result
//            ultralytics/utils/ops.py::process_mask（upsample=True 分支）
// =============================================================================
#include "yolov26_seg.hpp"

#include "trt_alpha/core/config.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"   // TRT_ALPHA_REGISTER_MODEL

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trt_alpha::seg {
namespace {

//! ultralytics NMS 的类别偏移基数（固定值，与 python 参考一致）。
constexpr float kMaxWh = 7680.0f;

}  // namespace

const std::string& YoloV26Seg::name() const noexcept
{
    return m_tag;
}

void YoloV26Seg::loadConfig(const core::ModelConfig& cfg)
{
    loadCommonConfig(cfg);

    // The fused preprocess applies 'scale' only; mean/std must not pass silently.
    core::requireScaleOnlyNorm(cfg, m_tag);

    if (m_numClass <= 0)
    {
        throw std::runtime_error(m_tag + ": num_class must be > 0 (got " +
                                 std::to_string(m_numClass) + ")");
    }

    // Mask coefficient count == prototype channel count. Both are 32 for every
    // yolo26-seg variant, but it is read (and cross-checked against output1's
    // real shape in discoverEngineIo) rather than hard-coded.
    const int extra = cfg.getInt("extra", 32);
    if (extra <= 0)
    {
        throw std::runtime_error(m_tag + ": extra (mask coefficients) must be > 0 (got " +
                                 std::to_string(extra) + ")");
    }
    m_numCoeffs = extra;

    if (m_topK <= 0)
    {
        throw std::runtime_error(m_tag + ": top_k must be > 0 (got " +
                                 std::to_string(m_topK) + ")");
    }

    // maskOrig = a full dstW->srcW bilinear resize per instance. On a 1912x1080
    // source that is ~616 of the ~1135 us each instance already costs, and the
    // built-in renderer never reads it (it rescales Segmentation::mask itself),
    // so it is opt-in. INI: [model] mask_orig = true
    m_emitMaskOrig = cfg.getBool("mask_orig", m_emitMaskOrig);

    TRT_LOG_INFO(m_tag << ": config num_class=" << m_numClass
                 << " num_coeffs=" << m_numCoeffs
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK
                 << " mask_orig=" << (m_emitMaskOrig ? "on" : "off"));
}

void YoloV26Seg::discoverEngineIo()
{
    const core::TensorDesc* input  = nullptr;
    const core::TensorDesc* dets   = nullptr;
    const core::TensorDesc* protos = nullptr;

    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput)
        {
            if (input == nullptr) { input = &t; }
            continue;
        }
        // Name first (the exported graph really is output0/output1), shape as a
        // fallback so a renamed graph still works: output0 is [B, C, anchors]
        // (3-D, thousands of anchors) and output1 is a 4-D prototype tensor.
        if (t.name == "output0")                              { dets   = &t; continue; }
        if (t.name == "output1")                              { protos = &t; continue; }
        if (t.shape.nbDims == 3 && dets == nullptr)           { dets   = &t; continue; }
        if (t.shape.nbDims == 4 && protos == nullptr)         { protos = &t; continue; }
    }
    if (input == nullptr || dets == nullptr || protos == nullptr)
    {
        throw std::runtime_error(
            m_tag + ": engine must expose one input plus two outputs "
            "(output0 = [B, 4+nc+coeffs, anchors], output1 = [B, coeffs, H, W])");
    }
    m_inputName  = input->name;
    m_outputName = dets->name;
    m_protoName  = protos->name;

    core::applyInputShape(*m_engine, m_inputName, core::Layout::NCHW, 3, m_cfg);

    const nvinfer1::Dims outDims = m_engine->contextShape(m_outputName);
    if (outDims.nbDims != 3)
    {
        throw std::runtime_error(m_tag + ": expect output0 as [batch, 4+nc+coeffs, anchors]");
    }
    if (outDims.d[0] > 0 && outDims.d[0] != m_cfg.batchSize)
    {
        throw std::runtime_error(
            m_tag + ": output0 batch = " + std::to_string(outDims.d[0]) +
            " but input batch = " + std::to_string(m_cfg.batchSize) +
            " (engine profile is the single source of truth; check 'batch_size')");
    }
    m_outC    = static_cast<int>(outDims.d[1]);
    m_anchors = static_cast<int>(outDims.d[2]);

    const int expectC = 4 + m_numClass + m_numCoeffs;
    if (m_outC != expectC)
    {
        throw std::runtime_error(
            m_tag + ": output0 channel = " + std::to_string(m_outC) +
            " but 4+num_class+extra = " + std::to_string(expectC) +
            " (check 'num_class' / 'extra' in INI)");
    }
    if (m_anchors <= 0)
    {
        throw std::runtime_error(m_tag + ": output0 anchors must be > 0 (got " +
                                 std::to_string(m_anchors) + ")");
    }

    const nvinfer1::Dims protoDims = m_engine->contextShape(m_protoName);
    if (protoDims.nbDims != 4)
    {
        throw std::runtime_error(m_tag + ": expect output1 as [batch, coeffs, H, W]");
    }
    if (protoDims.d[0] > 0 && protoDims.d[0] != m_cfg.batchSize)
    {
        throw std::runtime_error(
            m_tag + ": output1 batch = " + std::to_string(protoDims.d[0]) +
            " but input batch = " + std::to_string(m_cfg.batchSize));
    }
    if (protoDims.d[1] != m_numCoeffs)
    {
        throw std::runtime_error(
            m_tag + ": output1 channel = " + std::to_string(protoDims.d[1]) +
            " but output0 implies " + std::to_string(m_numCoeffs) + " mask coefficients");
    }
    m_protoH = static_cast<int>(protoDims.d[2]);
    m_protoW = static_cast<int>(protoDims.d[3]);
    if (m_protoH <= 0 || m_protoW <= 0)
    {
        throw std::runtime_error(m_tag + ": output1 spatial dims must be > 0");
    }
}

void YoloV26Seg::buildPlans()
{
    kernels::ops::ProprocessFusedVecYoloV26Config pcfg;
    pcfg.anchors    = m_anchors;
    pcfg.nc         = m_numClass;
    pcfg.extra      = m_numCoeffs;
    pcfg.maxBatch   = m_cfg.batchSize;
    pcfg.maxDet     = m_topK;
    pcfg.confThresh = m_confThreshold;
    pcfg.iouThresh  = m_iouThreshold;
    pcfg.maxWh      = kMaxWh;

    m_pp = std::make_unique<kernels::ops::CvProprocessFusedVecYoloV26Plan>(pcfg);
    if (!m_pp->ok())
    {
        throw std::runtime_error(m_tag + ": fused postprocess plan construction failed");
    }

    m_dot = std::make_unique<kernels::ops::CvMaskCoeffDotPlan>(
        m_numCoeffs, m_protoH, m_protoW);
    if (!m_dot->ok())
    {
        throw std::runtime_error(m_tag + ": mask coefficient dot plan construction failed");
    }

    // Stage 2 is "resize the whole prototype grid up to the network input size",
    // i.e. F.interpolate(logits, (dstH, dstW)) -- NOT a letterbox inverse. Feeding
    // it (dstW, dstH) as the target and (protoW, protoH) as the source makes
    // scale_mask compute gain = proto/dst and thus the identity crop, provided
    // the numbers line up; if they ever do not, the window would silently crop
    // real mask content away, so it is verified explicitly below.
    m_scale = std::make_unique<kernels::ops::CvScaleMaskPlan<float>>(
        m_cfg.dstW, m_cfg.dstH, m_protoW, m_protoH,
        kernels::ops::MaskInterp::Bilinear, /*padding=*/true);
    if (!m_scale->ok())
    {
        throw std::runtime_error(m_tag + ": mask upsample plan construction failed");
    }
    const kernels::ops::ScaleMaskInfo& si = m_scale->info();
    if (si.newW != m_protoW || si.newH != m_protoH ||
        si.cropLeft != 0 || si.cropTop != 0)
    {
        throw std::runtime_error(
            m_tag + ": prototype " + std::to_string(m_protoW) + "x" +
            std::to_string(m_protoH) + " does not upscale cleanly to " +
            std::to_string(m_cfg.dstW) + "x" + std::to_string(m_cfg.dstH) +
            " (crop window " + std::to_string(si.newW) + "x" +
            std::to_string(si.newH) + " at " + std::to_string(si.cropLeft) + "," +
            std::to_string(si.cropTop) + ") -- output1 must be a clean 1/N of the"
            " network input, otherwise the upscale would drop mask content");
    }

    // mask_orig needs the SOURCE image size, which is only known once the first
    // batch arrives -- its plan and scratch are built in setBatch() instead.
}

void YoloV26Seg::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;

    auto logBox = [&](const std::string& nm, int batch, int ch, int h, int w,
                      core::DataType dt, std::size_t bytes, core::MemorySpace sp)
    {
        core::detail::AllocInfo info;
        info.name     = nm.c_str();   // the caller keeps the string alive for this call
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
    logBox(m_tag + ".input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);

    m_inputNchw.allocate(static_cast<std::size_t>(B) * 3 * W * H * sizeof(float));
    logBox(m_tag + ".input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(static_cast<std::size_t>(B) * m_outC * m_anchors * sizeof(float));
    logBox(m_tag + ".output_src", B, m_outC, 1, m_anchors, core::DataType::Float32,
           m_outputSrc.bytes(), core::MemorySpace::Device);

    const std::size_t protoElems =
        static_cast<std::size_t>(m_numCoeffs) * m_protoH * m_protoW;
    m_outputProtos.allocate(static_cast<std::size_t>(B) * protoElems * sizeof(float));
    logBox(m_tag + ".output_protos", B, m_numCoeffs, m_protoH, m_protoW,
           core::DataType::Float32, m_outputProtos.bytes(), core::MemorySpace::Device);

    const int detStride = m_pp->detStride();
    const std::size_t detBytes =
        static_cast<std::size_t>(B) * m_topK * detStride * sizeof(float);
    const std::size_t cntBytes = static_cast<std::size_t>(B) * sizeof(int);

    m_dets.allocate(detBytes);
    logBox(m_tag + ".dets", B, m_topK, 1, detStride, core::DataType::Float32,
           m_dets.bytes(), core::MemorySpace::Device);

    m_counts.allocate(cntBytes);
    logBox(m_tag + ".counts", B, 1, 1, 1, core::DataType::Int32,
           m_counts.bytes(), core::MemorySpace::Device);

    m_detsHost.allocate(detBytes);
    logBox(m_tag + ".dets_host", B, m_topK, 1, detStride, core::DataType::Float32,
           m_detsHost.bytes(), core::MemorySpace::Host);

    m_countsHost.allocate(cntBytes);
    logBox(m_tag + ".counts_host", B, 1, 1, 1, core::DataType::Int32,
           m_countsHost.bytes(), core::MemorySpace::Host);

    const std::size_t logitElems = static_cast<std::size_t>(m_protoH) * m_protoW;
    m_logits.allocate(logitElems * kMaskChunk * sizeof(float));
    logBox(m_tag + ".logits", kMaskChunk, 1, m_protoH, m_protoW,
           core::DataType::Float32, m_logits.bytes(), core::MemorySpace::Device);

    const std::size_t maskBytes = static_cast<std::size_t>(W) * H * sizeof(float);
    m_maskUp.allocate(maskBytes * kMaskChunk);
    logBox(m_tag + ".mask_up", kMaskChunk, 1, H, W, core::DataType::Float32,
           m_maskUp.bytes(), core::MemorySpace::Device);

    // Stage 3 output lives on the device; only the finished uint8 mask is copied
    // back (4x less traffic than the float plane it used to ship home).
    const std::size_t maskU8Bytes = static_cast<std::size_t>(W) * H;
    m_maskU8.allocate(maskU8Bytes * kMaskChunk);
    logBox(m_tag + ".mask_u8", kMaskChunk, 1, H, W, core::DataType::UInt8,
           m_maskU8.bytes(), core::MemorySpace::Device);

    m_maskU8Host.allocate(maskU8Bytes * kMaskChunk);
    logBox(m_tag + ".mask_u8_host", kMaskChunk, 1, H, W, core::DataType::UInt8,
           m_maskU8Host.bytes(), core::MemorySpace::Host);

    m_maskCounts.allocate(sizeof(int) * kMaskChunk);
    logBox(m_tag + ".mask_counts", kMaskChunk, 1, 1, 1, core::DataType::Int32,
           m_maskCounts.bytes(), core::MemorySpace::Device);

    m_maskCountsHost.allocate(sizeof(int) * kMaskChunk);
    logBox(m_tag + ".mask_counts_host", kMaskChunk, 1, 1, 1, core::DataType::Int32,
           m_maskCountsHost.bytes(), core::MemorySpace::Host);

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_outputName.c_str(), m_outputSrc.data()) ||
        !ctx->setTensorAddress(m_protoName.c_str(), m_outputProtos.data()))
    {
        throw std::runtime_error(m_tag + ": setTensorAddress failed");
    }
}

void YoloV26Seg::init(const core::ModelConfig& cfg)
{
    TRT_LOG_DEBUG(m_tag << "::init: entering");
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    buildPlans();
    allocateBuffers();
    m_segmentations.assign(static_cast<std::size_t>(m_cfg.batchSize), {});

    TRT_LOG_INFO(m_tag << ": initialized (batch=" << m_cfg.batchSize
                 << ", outC=" << m_outC
                 << ", anchors=" << m_anchors
                 << ", proto=" << m_protoW << "x" << m_protoH
                 << ", dst=" << m_cfg.dstW << "x" << m_cfg.dstH << ")");
}

void YoloV26Seg::ensurePreprocessPlan()
{
    if (m_pre && m_pre->matches(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH))
    {
        m_lb = m_pre->info();
        return;
    }

    m_pre = std::make_unique<kernels::ops::CvPreprocessVecPlan>(
        kernels::ops::CvPreprocessVecPlan::byDivisor(
            m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH,
            static_cast<double>(m_padValue), m_normScale));
    if (!m_pre->ok())
    {
        throw std::runtime_error(m_tag + ": preprocess plan construction failed");
    }
    m_lb = m_pre->info();

    TRT_LOG_DEBUG(m_tag << ": letterbox rebuilt  " << m_srcW << "x" << m_srcH
                 << " -> new " << m_lb.newW << "x" << m_lb.newH
                 << " pad(l/t)=" << m_lb.padLeft << "/" << m_lb.padTop
                 << " ratio=" << m_lb.ratio);
}

void YoloV26Seg::ensureMaskOrigPlans()
{
    if (!m_emitMaskOrig)
    {
        return;
    }

    if (!m_origScale || !m_origScale->matches(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH))
    {
        // scale_masks(padding=True): drop the letterbox bars and bilinear-resample
        // onto the source image. Same ATen port stage 2 uses, run backwards.
        m_origScale = std::make_unique<kernels::ops::CvScaleMaskPlan<float>>(
            m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH,
            kernels::ops::MaskInterp::Bilinear, /*padding=*/true);
        if (!m_origScale->ok())
        {
            throw std::runtime_error(m_tag + ": mask_orig scale plan construction failed");
        }

        // The crop window must be the very letterbox the preprocess applied,
        // otherwise the mask is resampled off a different rectangle than the
        // image was padded with -- a silent misalignment nothing else catches.
        const kernels::ops::ScaleMaskInfo& oi = m_origScale->info();
        if (oi.newW != m_lb.newW || oi.newH != m_lb.newH ||
            oi.cropLeft != m_lb.padLeft || oi.cropTop != m_lb.padTop)
        {
            throw std::runtime_error(
                m_tag + ": mask_orig crop window " + std::to_string(oi.newW) + "x" +
                std::to_string(oi.newH) + " at " + std::to_string(oi.cropLeft) + "," +
                std::to_string(oi.cropTop) + " does not match the preprocess letterbox " +
                std::to_string(m_lb.newW) + "x" + std::to_string(m_lb.newH) + " at " +
                std::to_string(m_lb.padLeft) + "," + std::to_string(m_lb.padTop));
        }
    }

    const std::size_t origElems = static_cast<std::size_t>(m_srcW) * m_srcH;
    if (m_origF.bytes() < origElems * sizeof(float))
    {
        m_origF.allocate(origElems * sizeof(float));
        m_origU8.allocate(origElems);
        m_origU8Host.allocate(origElems);

        TRT_LOG_DEBUG(m_tag << ": mask_orig scratch " << m_srcW << "x" << m_srcH
                     << " (" << ((origElems * sizeof(float)) >> 20) << " MB f32)");
    }
}

void YoloV26Seg::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error(m_tag + ": empty batch");
    }

    m_batch = requireBatchCapacity(*this, batch, m_tag.c_str());
    m_srcH  = batch.views[0].height;
    m_srcW  = batch.views[0].width;
    if (m_srcW <= 0 || m_srcH <= 0)
    {
        throw std::runtime_error(m_tag + ": invalid source size " +
                                 std::to_string(m_srcW) + "x" + std::to_string(m_srcH));
    }

    ensurePreprocessPlan();
    ensureMaskOrigPlans();

    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error(m_tag + ": batch.buffer is null");
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

void YoloV26Seg::preprocess()
{
    if (!m_pre)
    {
        throw std::runtime_error(m_tag + ": preprocess called before setBatch");
    }
    m_pre->launch(m_stream.get(), m_batch,
                  static_cast<const unsigned char*>(m_inputSrc.data()),
                  m_inputNchw.asFloat());
    if (!m_pre->ok())
    {
        throw std::runtime_error(m_tag + ": preprocess kernel launch failed");
    }
}

void YoloV26Seg::infer()
{
    if (!m_engine->context()->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error(m_tag + ": enqueueV3 failed");
    }
}

void YoloV26Seg::postprocess()
{
    m_segmentations.assign(static_cast<std::size_t>(m_batch), {});

    m_pp->launch(m_stream.get(), m_batch, m_outputSrc.asFloat(),
                 static_cast<int*>(m_counts.data()), m_dets.asFloat());
    if (!m_pp->ok())
    {
        throw std::runtime_error(m_tag + ": fused postprocess kernel launch failed");
    }

    const int         stride   = m_pp->detStride();
    const std::size_t cntBytes = static_cast<std::size_t>(m_batch) * sizeof(int);
    const std::size_t detBytes =
        static_cast<std::size_t>(m_batch) * m_topK * stride * sizeof(float);

    cudaMemcpyAsync(m_countsHost.data(), m_counts.data(), cntBytes,
                    cudaMemcpyDeviceToHost, m_stream.get());
    cudaMemcpyAsync(m_detsHost.data(), m_dets.data(), detBytes,
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    const int*   counts = static_cast<const int*>(m_countsHost.data());
    const float* dets   = m_detsHost.asFloat();          // host copy
    const float* detsDev = m_dets.asFloat();             // device original

    const int   W = m_cfg.dstW;
    const int   H = m_cfg.dstH;
    const float srcW = static_cast<float>(m_srcW);
    const float srcH = static_cast<float>(m_srcH);

    const auto clipX = [srcW](float v) { return std::min(std::max(v, 0.f), srcW); };
    const auto clipY = [srcH](float v) { return std::min(std::max(v, 0.f), srcH); };

    const std::size_t logitElems = static_cast<std::size_t>(m_protoH) * m_protoW;
    const std::size_t protoElems =
        static_cast<std::size_t>(m_numCoeffs) * m_protoH * m_protoW;
    const std::size_t upElems = static_cast<std::size_t>(W) * H;

    float* logitsDev = m_logits.asFloat();
    float* maskUpDev = m_maskUp.asFloat();
    std::uint8_t* maskU8Dev  = static_cast<std::uint8_t*>(m_maskU8.data());
    std::uint8_t* maskU8Home = static_cast<std::uint8_t*>(m_maskU8Host.data());
    int*          maskCntDev = static_cast<int*>(m_maskCounts.data());
    const int*    maskCntHome = static_cast<const int*>(m_maskCountsHost.data());

    int produced = 0;
    int allZero  = 0;

    for (int b = 0; b < m_batch; ++b)
    {
        const int n = std::clamp(counts[b], 0, m_topK);
        if (n == 0) { continue; }

        const float* detRow   = dets + static_cast<std::size_t>(b) * m_topK * stride;
        const float* detsBase = detsDev + static_cast<std::size_t>(b) * m_topK * stride;
        const float* protosB  = m_outputProtos.asFloat()
                              + static_cast<std::size_t>(b) * protoElems;
        auto& out = m_segmentations[static_cast<std::size_t>(b)];
        out.reserve(out.size() + static_cast<std::size_t>(n));

        // Instances are batched in chunks of kMaskChunk: every dot / upsample
        // launch of a chunk is issued first, then one contiguous copy brings the
        // whole chunk back, then ONE synchronize() releases it to the CPU. That
        // turns n sync points per image into ceil(n / kMaskChunk).
        for (int base = 0; base < n; base += kMaskChunk)
        {
            const int m = std::min(kMaskChunk, n - base);

            for (int j = 0; j < m; ++j)
            {
                const float* coeffsDev = detsBase
                                       + static_cast<std::size_t>(base + j) * stride + 6;
                float* slotLogits = logitsDev + static_cast<std::size_t>(j) * logitElems;
                float* slotMaskUp = maskUpDev + static_cast<std::size_t>(j) * upElems;

                // ---- stage 1: logits[160x160] = coeffs . protos --------------
                m_dot->launch(m_stream.get(), coeffsDev, protosB, slotLogits);
                if (!m_dot->ok())
                {
                    throw std::runtime_error(m_tag + ": mask coefficient dot kernel failed");
                }

                // ---- stage 2: bilinear upsample to the network input size ----
                m_scale->launch(m_stream.get(), /*batch=*/1,
                                slotLogits, slotMaskUp, /*channels=*/1);
                if (!m_scale->ok())
                {
                    throw std::runtime_error(m_tag + ": mask upsample kernel failed");
                }
            }

            // ---- stage 3 on the DEVICE: binarize (> 0) + crop to the box -----
            // Pixel indices inside the half-open window
            //     [ceil(x1), ceil(x2)) x [ceil(y1), ceil(y2))
            // are the ones satisfying `x >= x1 && x < x2 && y >= y1 && y < y2`
            // (ultralytics crop_mask: right and bottom strict). Everything else
            // is 0. The same pass rewrites the up-slots to the 0/1 float form
            // that mask_orig interpolates, so no second plane pass is needed.
            cudaMemsetAsync(maskCntDev, 0,
                            sizeof(int) * static_cast<std::size_t>(m), m_stream.get());

            const cudaError_t bErr = kernels::ops::binarizeCropMasks(
                m_stream.get(), m,
                maskUpDev, upElems,
                detsBase + static_cast<std::size_t>(base) * stride,
                stride, W, H, maskU8Dev, maskCntDev, /*thr=*/0.0f);
            if (bErr != cudaSuccess)
            {
                throw std::runtime_error(m_tag + ": mask binarize/crop kernel failed: " +
                                         cudaGetErrorString(bErr));
            }

            // Slots are contiguous on both sides, so this is one single copy --
            // uint8 now, i.e. a quarter of the float plane this used to ship home.
            const cudaError_t copyErr =
                cudaMemcpyAsync(maskU8Home, maskU8Dev,
                                upElems * static_cast<std::size_t>(m),
                                cudaMemcpyDeviceToHost, m_stream.get());
            const cudaError_t cntErr =
                cudaMemcpyAsync(m_maskCountsHost.data(), maskCntDev,
                                sizeof(int) * static_cast<std::size_t>(m),
                                cudaMemcpyDeviceToHost, m_stream.get());
            if (copyErr != cudaSuccess || cntErr != cudaSuccess)
            {
                throw std::runtime_error(
                    m_tag + ": mask copy back failed: " +
                    cudaGetErrorString(copyErr != cudaSuccess ? copyErr : cntErr));
            }
            m_stream.synchronize();

            for (int j = 0; j < m; ++j)
            {
                const float* d = detRow + static_cast<std::size_t>(base + j) * stride;

                // Official `keep = masks.amax((-2,-1)) > 0`: an instance whose mask
                // is empty after cropping is dropped from the result list entirely.
                // The count came from the kernel -- nothing is scanned on the host.
                if (maskCntHome[j] == 0)
                {
                    ++allZero;
                    continue;
                }

                auto maskBuf = core::Buffer::createHost(W, H, 1, core::DataType::UInt8);
                std::memcpy(maskBuf->mutableData(),
                            maskU8Home + static_cast<std::size_t>(j) * upElems,
                            upElems);

                Segmentation s;
                s.box.left       = clipX(m_lb.toOrigX(d[0]));
                s.box.top        = clipY(m_lb.toOrigY(d[1]));
                s.box.right      = clipX(m_lb.toOrigX(d[2]));
                s.box.bottom     = clipY(m_lb.toOrigY(d[3]));
                s.box.confidence = d[4];
                s.box.label      = static_cast<int>(d[5]);
                s.mask      = maskBuf->view();   // network space (640x640, letterboxed)
                s.maskOwner = maskBuf;

                // ---- stage 4: un-letterbox the mask to the SOURCE-IMAGE size --
                // Drop the letterbox bars and bilinear-resample onto srcW x srcH,
                // then `> 0.5` -- i.e. ops.scale_masks(mode="bilinear") plus the
                // comparison that follows it in the python reference. The source
                // is the 0/1 float mask the binarize kernel left in the up-slot,
                // which is exactly the python bool mask cast to float, and the
                // resampler is the same ATen bilinear port stage 2 uses.
                //
                // Off unless [model] mask_orig = true: it costs a full-resolution
                // resize per instance and the built-in renderer never reads the
                // result (it rescales `mask` itself). One instance at a time, so
                // the float scratch stays at a single srcW*srcH plane.
                if (m_emitMaskOrig)
                {
                    const std::size_t origElems =
                        static_cast<std::size_t>(m_srcW) * m_srcH;

                    m_origScale->launch(m_stream.get(), /*batch=*/1,
                                        maskUpDev + static_cast<std::size_t>(j) * upElems,
                                        m_origF.asFloat(), /*channels=*/1);
                    if (!m_origScale->ok())
                    {
                        throw std::runtime_error(m_tag + ": mask_orig scale kernel failed");
                    }

                    const cudaError_t tErr = kernels::ops::thresholdMaskToU8(
                        m_stream.get(), m_origF.asFloat(),
                        static_cast<std::uint8_t*>(m_origU8.data()),
                        origElems, /*thr=*/0.5f);
                    if (tErr != cudaSuccess)
                    {
                        throw std::runtime_error(
                            m_tag + ": mask_orig threshold kernel failed: " +
                            cudaGetErrorString(tErr));
                    }

                    cudaMemcpyAsync(m_origU8Host.data(), m_origU8.data(), origElems,
                                    cudaMemcpyDeviceToHost, m_stream.get());
                    m_stream.synchronize();

                    auto fullBuf = core::Buffer::createHost(
                        m_srcW, m_srcH, 1, core::DataType::UInt8);
                    std::memcpy(fullBuf->mutableData(), m_origU8Host.data(), origElems);
                    s.maskOrig      = fullBuf->view();
                    s.maskOrigOwner = fullBuf;
                }

                out.push_back(std::move(s));
                ++produced;
            }
        }
    }

    TRT_LOG_DEBUG(m_tag << "::postprocess: batch=" << m_batch
                 << " masks=" << produced
                 << " dropped(all-zero)=" << allZero);
}

void YoloV26Seg::reset()
{
    for (auto& v : m_segmentations) { v.clear(); }
    m_batch = 0;
}

}  // namespace trt_alpha::seg

TRT_ALPHA_REGISTER_MODEL("yolov26_seg", trt_alpha::seg::YoloV26Seg);
