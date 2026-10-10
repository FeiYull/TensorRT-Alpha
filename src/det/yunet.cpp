// =============================================================================
//  trt_alpha :: det :: YuNet（实现）
// =============================================================================
#include "yunet.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/kernels/cast.hpp"
#include "trt_alpha/kernels/legacy/preprocess.hpp"

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

//! 三个输出的行宽：定义在 kernels/legacy/postprocess.hpp（kernel 与形状校验共用的单一来源）。
using kernels::kYuNetConfRow;
using kernels::kYuNetIouRow;
using kernels::kYuNetLocRow;
using kernels::kYuNetObjectsRow;

//! nvinfer1::Dims → "[B, N, C]"（异常信息用）。
std::string shapeOf(const nvinfer1::Dims& d)
{
    if (d.nbDims <= 0) { return "scalar"; }
    std::string s = "[";
    for (int i = 0; i < d.nbDims; ++i)
    {
        if (i > 0) { s += ", "; }
        s += std::to_string(d.d[i]);
    }
    return s + "]";
}

//! 抛输出形状不符。
[[noreturn]] void failShape(const std::string& name, const std::string& why,
                            const nvinfer1::Dims& d)
{
    throw std::runtime_error("yunet: output '" + name + "' " + why +
                             " (engine shape " + shapeOf(d) + ")");
}

//! 校验一个输出的 [B, N, C] 契约，返回引擎声明的 N。
//! N / 行宽 / batch 任一不符都会让 kernel 越界读写 —— 宁可报错，不静默跑。
//! 形状来源是 context（已下发输入形状后）的**推导值**，即引擎真相。
int checkOutputShape(const nvinfer1::IExecutionContext& ctx, const std::string& name,
                     int batch, int rowWidth)
{
    const nvinfer1::Dims d = ctx.getTensorShape(name.c_str());

    if (d.nbDims != 3)
    {
        failShape(name, "expected rank 3 [B, N, C]", d);
    }
    if (d.d[2] != rowWidth)
    {
        failShape(name, "row width " + std::to_string(d.d[2]) +
                        " != expected " + std::to_string(rowWidth), d);
    }
    if (d.d[1] <= 0)
    {
        failShape(name, "candidate count is not statically derivable", d);
    }
    if (d.d[0] > 0 && d.d[0] != batch)
    {
        failShape(name, "batch " + std::to_string(d.d[0]) +
                        " != " + std::to_string(batch), d);
    }
    return d.d[1];
}

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

    out[0]  = float(p1_h); out[1]  = float(p1_w); out[2]  = 51.f;
    out[3]  = float(p2_h); out[4]  = float(p2_w); out[5]  = 34.f;
    out[6]  = float(p3_h); out[7]  = float(p3_w); out[8]  = 34.f;
    out[9]  = float(p4_h); out[10] = float(p4_w); out[11] = 51.f;
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
    loadCommonConfig(cfg);

    // YuNet 官方默认值
    if (cfg.getString("conf_thresh", "").empty()) m_confThreshold = 0.3f;
    if (cfg.getString("top_k",       "").empty()) m_topK          = 1000;

    m_objectsRow = kernels::kYuNetObjectsRow;

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

    // 输入结构守卫：秩 / 通道轴 / 物理格式。
    // H/W 是动态维且由原图尺寸决定（不 resize，见 rebuildForSize），故不解析配置意图值。
    const core::TensorDesc* in = m_engine->find(m_inputName);
    if (in == nullptr)
    {
        throw std::runtime_error("yunet: input tensor '" + m_inputName + "' not found");
    }
    core::validateInputTensor(*in, core::Layout::NCHW, 3, "YuNet");

    // batch 与其余模型同口径（引擎 profile 为唯一真相源，不符即抛）。
    // YuNet 的 H/W 取自原图、不走 core::applyInputShape，故在此单独落定。
    m_cfg.batchSize = core::resolveBatch(*in, m_cfg.batchSize, "YuNet",
                                         m_cfg.maxBatchSize).batch;

    TRT_LOG_INFO("YuNet: io names: input='" << m_inputName
                 << "' loc='" << m_locName
                 << "' conf='" << m_confName
                 << "' iou='" << m_iouName << "'");
}

void YuNet::allocateConstBuffers()
{
    m_minSizes.allocate(sizeof(kMinSizesHost));
    cudaMemcpyAsync(m_minSizes.data(), kMinSizesHost, sizeof(kMinSizesHost),
                    cudaMemcpyHostToDevice, m_stream.get());
    m_variances.allocate(sizeof(kVariancesHost));
    cudaMemcpyAsync(m_variances.data(), kVariancesHost, sizeof(kVariancesHost),
                    cudaMemcpyHostToDevice, m_stream.get());
    m_featHw.allocate(4 * 3 * sizeof(float));
    m_stream.synchronize();
}

void YuNet::rebuildForSize(int W, int H)
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    const core::TensorDesc* in = m_engine->find(m_inputName);

    // ---- 1. 下发输入形状 ----
    // 引擎 H/W 为动态维时下发原图尺寸；静态维时原图尺寸必须与引擎一致
    // （YuNet 不 resize，直接喂原图，尺寸不符只能报错）。
    const bool dynHw = (in != nullptr) && (in->shape.nbDims >= 4) &&
                       (in->shape.d[2] < 0 || in->shape.d[3] < 0);
    if (dynHw)
    {
        if (!ctx->setInputShape(m_inputName.c_str(), nvinfer1::Dims4(m_batch, 3, H, W)))
        {
            throw std::runtime_error(
                "yunet: setInputShape failed for " + std::to_string(m_batch) + "x3x" +
                std::to_string(H) + "x" + std::to_string(W) +
                " (outside engine profile range?)");
        }
    }
    else
    {
        if (in == nullptr || in->shape.d[2] != H || in->shape.d[3] != W)
        {
            throw std::runtime_error(
                "yunet: input " + std::to_string(W) + "x" + std::to_string(H) +
                " != engine " + (in ? shapeOf(in->shape) : std::string("(unknown)")) +
                " (engine H/W are static dims and YuNet does no resize)");
        }
    }

    // ---- 2. 输出形状守门（引擎声明为唯一真相源）----
    // loc/conf/iou 均为 [B, N, C]；N 以 loc 为准，conf / iou 必须与之一致。
    const int nLoc  = checkOutputShape(*ctx, m_locName,  m_batch, kYuNetLocRow);
    const int nConf = checkOutputShape(*ctx, m_confName, m_batch, kYuNetConfRow);
    const int nIou  = checkOutputShape(*ctx, m_iouName,  m_batch, kYuNetIouRow);
    if (nConf != nLoc || nIou != nLoc)
    {
        throw std::runtime_error("yunet: loc/conf/iou candidate count mismatch (" +
                                 std::to_string(nLoc) + " / " + std::to_string(nConf) +
                                 " / " + std::to_string(nIou) + ")");
    }

    // ---- 3. 先验框（Host 公式）：其候选数必须与引擎一致 ----
    // 先验框由 kMinSizes / kSteps / feature-map 公式决定；引擎 N 与它不符
    // 说明 kernel 假设与该模型不匹配 —— 无解，直接报错（不静默跑出错误框）。
    std::vector<float> featHw(4 * 3);
    calFeatureMapSize(W, H, featHw.data());
    const int formulaN = calNumCandidates(featHw.data());
    if (formulaN <= 0)
    {
        throw std::runtime_error("yunet: numCandidates <= 0 at input size " +
                                 std::to_string(W) + "x" + std::to_string(H));
    }
    if (formulaN != nLoc)
    {
        throw std::runtime_error(
            "yunet: engine declares N=" + std::to_string(nLoc) +
            " but prior-box formula gives N=" + std::to_string(formulaN) +
            " at " + std::to_string(W) + "x" + std::to_string(H) +
            " (kernel assumptions do not match this engine)");
    }
    m_numCandidates = nLoc;

    // ---- 4. 分配 / 上传（尺寸全部以引擎确认的 N 为准）----
    cudaMemcpyAsync(m_featHw.data(), featHw.data(), featHw.size() * sizeof(float),
                    cudaMemcpyHostToDevice, m_stream.get());

    std::vector<float> priorBoxes(std::size_t(m_numCandidates) * 4);
    calPriorBox(featHw.data(), W, H, priorBoxes.data());
    m_priorBoxes.allocate(priorBoxes.size() * sizeof(float));
    cudaMemcpyAsync(m_priorBoxes.data(), priorBoxes.data(),
                    priorBoxes.size() * sizeof(float),
                    cudaMemcpyHostToDevice, m_stream.get());

    m_inputNchw.allocate(std::size_t(m_batch) * 3 * m_srcH * m_srcW * sizeof(float));

    const std::size_t cand = std::size_t(m_batch) * std::size_t(m_numCandidates);
    m_outputLoc.allocate(cand * kYuNetLocRow * sizeof(float));
    m_outputConf.allocate(cand * kYuNetConfRow * sizeof(float));
    m_outputIou.allocate(cand * kYuNetIouRow * sizeof(float));

    const std::size_t objectsPerImage = 1 + std::size_t(m_topK) * m_objectsRow;
    m_objects.allocate(std::size_t(m_batch) * objectsPerImage * sizeof(float));
    m_objectsHost.allocate(std::size_t(m_batch) * objectsPerImage * sizeof(float));

    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_locName.c_str(), m_outputLoc.data()) ||
        !ctx->setTensorAddress(m_confName.c_str(), m_outputConf.data()) ||
        !ctx->setTensorAddress(m_iouName.c_str(), m_outputIou.data()))
    {
        throw std::runtime_error("yunet: setTensorAddress failed");
    }
    m_stream.synchronize();

    TRT_LOG_INFO("YuNet: resized to " << W << "x" << H
                 << " -> numCandidates=" << m_numCandidates
                 << " (engine-confirmed)");
}

void YuNet::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = cfg.sharedEngine
        ? std::make_unique<core::TrtEngine>(cfg.sharedEngine)
        : std::make_unique<core::TrtEngine>(cfg.engine);
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

    // 输入 HWC buffer（用 size_t 计算，避免大分辨率 × 大 batch 时 int 溢出）
    const std::size_t needF32 =
        std::size_t(m_batch) * m_srcH * m_srcW * 3 * sizeof(float);
    if (m_inputHwc.bytes() < needF32)
    {
        m_inputHwc.allocate(needF32);
    }
    rebuildForSize(m_srcW, m_srcH);

    // batch.buffer 是 uint8：H2D uint8 -> GPU kernel -> float HWC
    const std::size_t totalU8 = std::size_t(m_srcH) * m_srcW * 3 * m_batch;
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yunet: batch.buffer is null");
    }
    if (m_inputU8.bytes() < totalU8)
    {
        m_inputU8.allocate(totalU8);
    }
    cudaMemcpyAsync(m_inputU8.data(), batch.buffer->data(), totalU8,
                    cudaMemcpyHostToDevice, m_stream.get());
    kernels::u8ToF32(m_stream.get(),
                     static_cast<const std::uint8_t*>(m_inputU8.data()),
                     m_inputHwc.asFloat(),
                     totalU8);
    m_stream.synchronize();
}

void YuNet::preprocess()
{
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