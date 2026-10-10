// =============================================================================
//  trt_alpha :: seg :: YoloV8Seg（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOv8-seg 实例分割。
//  - 2 个输出：output0 [B, 116, 8400] / output1 [B, 32, 160, 160]
//  - output0: 4 + 80 + 32（前 4 是 xywh，中 80 是 class，后 32 是 mask 系数）
//  - output1: 32 个 160×160 的 mask 原型
//  - mask 后处理在 CPU（cv::Mat，不用 Eigen）
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/kernels/legacy/postprocess.hpp"
#include "trt_alpha/kernels/legacy/preprocess.hpp"
#include "trt_alpha/seg/segmentor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::seg {

class YoloV8Seg final : public seg::ISegmentor
{
public:
    YoloV8Seg() = default;
    ~YoloV8Seg() override = default;

    YoloV8Seg(const YoloV8Seg&) = delete;
    YoloV8Seg& operator=(const YoloV8Seg&) = delete;

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
    // core::ModelConfig m_cfg;
    // int m_numClass = 80;
    int m_numMaskCoeffs = 32;
    int m_maskProtoH = 160;
    int m_maskProtoW = 160;
    // float m_confThreshold = 0.25f;
    // float m_iouThreshold = 0.7f;
    // int m_topK = 300;
    // float m_normScale = 255.f;
    // float m_normMean[3] = {0.f, 0.f, 0.f};
    // float m_normStd[3]  = {1.f, 1.f, 1.f};
    // float m_padValue = 114.f;

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_output0Name;  // "output0"
    std::string m_output1Name;  // "output1"

    int m_batch = 0;
    int m_srcW = 0;
    int m_srcH = 0;

    int m_srcRow = 0;      // 4 + nc + 32
    int m_anchors = 0;     // 8400

    core::CudaStream m_stream;
    core::DeviceBuffer m_inputSrc;
    core::DeviceBuffer m_resizeOut;
    core::DeviceBuffer m_inputNchw;
    core::DeviceBuffer m_outputSrc;         // [B, 116, 8400]
    core::DeviceBuffer m_outputTransposed;  // [B, 8400, 116]
    core::DeviceBuffer m_outputSeg;         // [B, 32, 160, 160]
    core::DeviceBuffer m_objects;           // [B, 1 + topK*39]
    core::PinnedBuffer m_objectsHost;
    core::PinnedBuffer m_outputSegHost;

    int m_objectsPerImage = 0;
    int m_objectsRow = 0;  // 7 + 32

    trt_alpha::kernels::AffineMat m_dst2src{};

    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
};

}  // namespace trt_alpha::seg