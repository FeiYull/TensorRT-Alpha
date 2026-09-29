// =============================================================================
//  trt_alpha :: det :: U2Net（私有头文件）
// -----------------------------------------------------------------------------
//  U-2-Net 显著性目标检测（salient object detection）。
//  - 输入 [B, 3, 320, 320]
//  - 输出 [B, 1, 320, 320]（单通道显著性概率）
//  - 预处理：BGR->RGB + resize + divByMax + ImageNet 归一化 + HWC->CHW
//  - 后处理：min-max 归一化到 [0, 255] + resize 回原图 + 转 uint8
//  - mask 是整图，box 填整图占位（TODO: seg 支持无框分割）
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/kernels/postprocess.hpp"
#include "trt_alpha/kernels/preprocess.hpp"
#include "trt_alpha/seg/segmentor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::det {

class U2Net final : public seg::ISegmentor
{
public:
    U2Net() = default;
    ~U2Net() override = default;

    U2Net(const U2Net&) = delete;
    U2Net& operator=(const U2Net&) = delete;

    [[nodiscard]] const std::string& name() const noexcept override;

    void init(const core::ModelConfig& cfg) override;
    void setBatch(const core::Batch& batch) override;
    void preprocess() override;
    void infer() override;
    void postprocess() override;
    void reset() override;

    [[nodiscard]] const std::vector<core::TensorDesc>& describe() const noexcept override
    {
        static const std::vector<core::TensorDesc> kEmpty;
        return m_engine ? m_engine->ioTensors() : kEmpty;
    }

private:
    core::ModelConfig m_cfg;
    int m_numClass = 1;          // 显著性，单类
    float m_normScale = 1.f;     // u2net 的 scale = 1.0（预处理）
    float m_normMean[3] = {0.485f, 0.456f, 0.406f};
    float m_normStd[3]  = {0.229f, 0.224f, 0.225f};
    float m_postScale = 255.f;   // 后处理归一化到 [0, 255]

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output"

    int m_batch = 0;
    int m_srcW = 0;
    int m_srcH = 0;

    core::CudaStream m_stream;

    // 预处理
    core::DeviceBuffer m_inputSrc;      // [B, 3, srcH, srcW] float（uint8->float 后）
    core::DeviceBuffer m_inputRgb;      // [B, 3, srcH, srcW] float（BGR->RGB 后）
    core::DeviceBuffer m_inputResize;   // [B, 3, dstH, dstW] float（resize 后）
    core::DeviceBuffer m_inputNorm;     // [B, 3, dstH, dstW] float（归一化后）
    core::DeviceBuffer m_inputNchw;     // [B, 3, dstH, dstW] float（NCHW）

    // 预处理：max（每张图 RGB 最大值）
    core::DeviceBuffer m_maxValDevice;  // [B]

    // 后处理：min/max
    core::DeviceBuffer m_minValDevice;  // [B]
    core::DeviceBuffer m_postMaxDevice; // [B]

    // 输出
    core::DeviceBuffer m_outputSrc;     // [B, 1, dstH, dstW] float
    core::DeviceBuffer m_outputResize;  // [B, 1, srcH, srcW] float

    // Host（D2H 后转 uint8）
    core::PinnedBuffer m_outputResizeHost;  // [B, 1, srcH, srcW] float
    core::PinnedBuffer m_maskHost;          // [srcH, srcW] float

    trt_alpha::kernels::AffineMat m_dst2src{};
    trt_alpha::kernels::AffineMat m_src2dst{};

    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
};

}  // namespace trt_alpha::det