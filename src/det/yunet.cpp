// =============================================================================
//  trt_alpha :: det :: YuNet（实现）
// =============================================================================
#include "trt_alpha/kernels/preprocess.hpp"
#include "yunet.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_alpha::det {
namespace {

//! 常量（照 legacy libfacedetection.h）。
const float kMinSizesHost[4 * 3] = {
    10.f, 16.f, 24.f,
    32.f, 48.f, FLT_MAX,
    64.f, 96.f, FLT_MAX,
    128.f, 192.f, 256.f
};
const int kMinSizesDim[4] = { 3, 2, 2, 3 };
const float kSteps[4] = { 8.f, 16.f, 32.f, 64.f };
const float kVariancesHost[2] = { 0.1f, 0.2f };

//! 计算 4 个尺度的 feature map 尺寸（照 legacy calFeatureMapSize）。
void calFeatureMapSize(int srcW, int srcH, float* out /* [4*3] */)
{
    const int h0 = int(int((srcH + 1) / 2) / 2);
    const int w0 = int(int((srcW + 1) / 2) / 2);

    const int p1_h = int(h0 / 2);
    const int p1_w = int(w0 / 2);
    const int p2_h = int(p1_h / 2);
    const int p2_w = int(p1_w / 2);
    const int p3_h = int(p2_h / 2);
    const int p3_w = int(p2_w / 2);
    const int p4_h = int(p3_h / 2);
    const int p4_w = int(p3_w / 2);

    out[0]  = float(p1_h); out[1]  = float(p1_w); out[2]  = 51.f;   // P1: 51 通道
    out[3]  = float(p2_h); out[4]  = float(p2_w); out[5]  = 34.f;   // P2: 34 通道
    out[6]  = float(p3_h); out[7]  = float(p3_w); out[8]  = 34.f;   // P3: 34 通道
    out[9]  = float(p4_h); out[10] = float(p4_w); out[11] = 51.f;   // P4: 51 通道
}

//! 计算 prior boxes（照 legacy calPriorBox）。
void calPriorBox(const float* featHw, int srcW, int srcH, float* out /* [N, 4] */)
{
    int idx = 0;
    for (int k = 0; k < 4; ++k)
    {
        const int fh = int(featHw[k * 3 + 0]);
        const int fw = int(featHw[k * 3 + 1]);
        for (int i = 0; i < fh; ++i)
        {
            for (int j = 0; j < fw; ++j)
            {
                for (int m = 0; m < kMinSizesDim[k]; ++m)
                {
                    out[idx++] = (float(j) + 0.5f) * kSteps[k] / float(srcW);
                    out[idx++] = (float(i) + 0.5f) * kSteps[k] / float(srcH);
                    out[idx++] = kMinSizesHost[k * 3 + m] / float(srcW);
                    out[idx++] = kMinSizesHost[k * 3 + m] / float(srcH);
                }
            }
        }
    }
}

//! 计算候选框总数 N。
int calNumCandidates(const float* featHw)
{
    int n = 0;
    for (int k = 0; k < 4; ++k)
    {
        const int fh = int(featHw[k * 3 + 0]);
        const int fw = int(featHw[k * 3 + 1]);
        n += fh * fw * kMinSizesDim[k];
    }
    return n;
}

}  // namespace

const std::string& YuNet::name() const noexcept
{
    static const std::string kName = "yunet";
    return kName;
}

void YuNet::loadConfig(const core::ModelConfig& cfg)
{
    m_cfg = cfg;
    m_numClass = cfg.getInt("num_class", 2);
    m_confThreshold = cfg.getFloat("conf_thresh", 0.3f);
    m_iouThreshold = cfg.getFloat("iou_thresh", 0.45f);
    m_topK = cfg.getInt("top_k", 1000);

    TRT_LOG_INFO("YuNet: config num_class=" << m_numClass
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK);
}

void YuNet::discoverEngineIo()
{
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput)                 { m_inputName = t.name; continue; }
        if (t.name == "loc")           { m_locName = t.name;  continue; }
        if (t.name == "conf")          { m_confName = t.name; continue; }
        if (t.name == "iou")           { m_iouName = t.name;  continue; }
    }
    if (m_inputName.empty() || m_locName.empty() ||
        m_confName.empty() || m_iouName.empty())
    {
        throw std::runtime_error("yunet: missing expected I/O tensors");
    }
    TRT_LOG_INFO("YuNet: io names: input='" << m_inputName
                 << "' loc='" << m_locName
                 << "' conf='" << m_confName
                 << "' iou='" << m_iouName << "'");
}

void YuNet::allocateConstBuffers()
{
    // min_sizes
    m_minSizes.allocate(sizeof(kMinSizesHost));
    cudaMemcpyAsync(m_minSizes.data(), kMinSizesHost, sizeof(kMinSizesHost),
                    cudaMemcpyHostToDevice, m_stream.get());
    // variances
    m_variances.allocate(sizeof(kVariancesHost));
    cudaMemcpyAsync(m_variances.data(), kVariancesHost, sizeof(kVariancesHost),
                    cudaMemcpyHostToDevice, m_stream.get());
    // feat_hw（下一行 setBatch 里填）
    m_featHw.allocate(4 * 3 * sizeof(float));
    m_stream.synchronize();
}

void YuNet::rebuildForSize(int W, int H)
{
    // 算 feature map 尺寸
    std::vector<float> featHw(4 * 3);
    calFeatureMapSize(W, H, featHw.data());
    cudaMemcpyAsync(m_featHw.data(), featHw.data(), featHw.size() * sizeof(float),
                    cudaMemcpyHostToDevice, m_stream.get());

    // 算 N
    m_numCandidates = calNumCandidates(featHw.data());
    if (m_numCandidates <= 0)
    {
        throw std::runtime_error("yunet: numCandidates <= 0, invalid input size");
    }

    // 算 prior boxes
    std::vector<float> priorBoxes(std::size_t(m_numCandidates) * 4);
    calPriorBox(featHw.data(), W, H, priorBoxes.data());
    m_priorBoxes.allocate(priorBoxes.size() * sizeof(float));
    cudaMemcpyAsync(m_priorBoxes.data(), priorBoxes.data(),
                    priorBoxes.size() * sizeof(float),
                    cudaMemcpyHostToDevice, m_stream.get());

    // 输入 NCHW buffer（HWC->CHW 后的结果）
    m_inputNchw.allocate(std::size_t(m_batch) * 3 * m_srcH * m_srcW * sizeof(float));

    // 输出 buffer
    m_outputLoc.allocate(std::size_t(m_batch) * m_numCandidates * 14 * sizeof(float));
    m_outputConf.allocate(std::size_t(m_batch) * m_numCandidates * 2 * sizeof(float));
    m_outputIou.allocate(std::size_t(m_batch) * m_numCandidates * 1 * sizeof(float));

    const std::size_t objectsPerImage = 1 + std::size_t(m_topK) * m_objectsRow;
    m_objects.allocate(std::size_t(m_batch) * objectsPerImage * sizeof(float));
    m_objectsHost.allocate(std::size_t(m_batch) * objectsPerImage * sizeof(float));

    // 绑定
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    ctx->setInputShape(m_inputName.c_str(), nvinfer1::Dims4(m_batch, 3, H, W));
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_locName.c_str(), m_outputLoc.data()) ||
        !ctx->setTensorAddress(m_confName.c_str(), m_outputConf.data()) ||
        !ctx->setTensorAddress(m_iouName.c_str(), m_outputIou.data()))
    {
        throw std::runtime_error("yunet: setTensorAddress failed");
    }
    m_stream.synchronize();

    TRT_LOG_INFO("YuNet: resized to " << W << "x" << H
                 << " -> numCandidates=" << m_numCandidates);
}

void YuNet::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateConstBuffers();
    m_detections.assign(static_cast<std::size_t>(m_cfg.batchSize), {});
    TRT_LOG_INFO("YuNet: initialized (batch=" << m_cfg.batchSize << ")");
}

void YuNet::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yunet: empty batch");
    }
    m_batch = static_cast<int>(batch.views.size());
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    // 输入 HWC buffer
    if (static_cast<int>(m_inputHwc.bytes()) <
        m_batch * m_srcH * m_srcW * 3 * static_cast<int>(sizeof(float)))
    {
        m_inputHwc.allocate(std::size_t(m_batch) * m_srcH * m_srcW * 3 * sizeof(float));
    }
    rebuildForSize(m_srcW, m_srcH);

    // 把 batch.buffer 的 uint8 转 float 上传
    const std::size_t totalU8 = std::size_t(m_srcH) * m_srcW * 3 * m_batch;
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yunet: batch.buffer is null");
    }
    std::vector<float> hostF32(totalU8);
    const std::uint8_t* srcU8 = batch.buffer->data();
    for (std::size_t i = 0; i < totalU8; ++i)
    {
        hostF32[i] = static_cast<float>(srcU8[i]);
    }
    cudaMemcpyAsync(m_inputHwc.data(), hostF32.data(), totalU8 * sizeof(float),
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();
}

void YuNet::preprocess()
{
    // HWC float (m_inputHwc) -> CHW float (m_inputNchw)
    kernels::hwcToChw(m_stream.get(), m_batch,
                      m_inputHwc.asFloat(),
                      m_inputNchw.asFloat(),
                      m_srcW, m_srcH);
}

void YuNet::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yunet: enqueueV3 failed");
    }
}

void YuNet::postprocess()
{
    m_detections.assign(static_cast<std::size_t>(m_batch), {});

    cudaMemsetAsync(m_objects.data(), 0,
                    std::size_t(m_batch) * (1 + std::size_t(m_topK) * m_objectsRow) * sizeof(float),
                    m_stream.get());

    kernels::decodeYuNetHead(m_stream.get(),
                             m_outputLoc.asFloat(), m_outputConf.asFloat(),
                             m_outputIou.asFloat(),
                             m_batch, m_numCandidates,
                             m_srcW, m_srcH,
                             m_confThreshold, m_topK,
                             m_priorBoxes.asFloat(),
                             m_variances.asFloat(),
                             m_objects.asFloat());

    // NMS（只处理前 7 个字段，关键点不参与）
    kernels::YoloDecodeParams p;
    p.batch = m_batch;
    p.topK = m_topK;
    p.iouThreshold = m_iouThreshold;
    kernels::nmsFast(m_stream.get(), p, m_objects.asFloat(), m_objectsRow);

    cudaMemcpyAsync(m_objectsHost.data(), m_objects.data(),
                    std::size_t(m_batch) * (1 + std::size_t(m_topK) * m_objectsRow) * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    const float* host = m_objectsHost.asFloat();
    for (int b = 0; b < m_batch; ++b)
    {
        const float* row = host + std::size_t(b) * (1 + std::size_t(m_topK) * m_objectsRow);
        const int count = std::clamp(static_cast<int>(row[0]), 0, m_topK);
        for (int i = 0; i < count; ++i)
        {
            const float* o = row + 1 + i * m_objectsRow;
            if (o[6] < 0.5f) { continue; }
            Detection d;
            d.left   = o[0];
            d.top    = o[1];
            d.right  = o[2];
            d.bottom = o[3];
            d.confidence = o[4];
            d.label = static_cast<int>(o[5]);
            // 5 个关键点
            for (int k = 0; k < 5; ++k)
            {
                Point2f pt;
                pt.x = o[7 + k * 2 + 0];
                pt.y = o[7 + k * 2 + 1];
                d.land_marks.push_back(pt);
            }
            m_detections[std::size_t(b)].push_back(d);
        }
    }
}

void YuNet::reset()
{
    for (auto& v : m_detections) { v.clear(); }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("yunet", trt_alpha::det::YuNet);