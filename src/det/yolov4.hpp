// =============================================================================
//  trt_alpha :: det :: YoloV4（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOv4 检测模型。
//  输出 [B, anchors, 1, 4+nc]（4 维，无 objectness，cx/cy/w/h 归一化 0~1），
//  decode 用 kernels::decodeYoloV4Head。
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/det/detector.hpp"
#include "trt_alpha/kernels/postprocess.hpp"
#include "trt_alpha/kernels/preprocess.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::det {

//! YOLOv4 检测模型。
class YoloV4 final : public IDetector
{
public:
    YoloV4() = default;
    ~YoloV4() override = default;

    YoloV4(const YoloV4&) = delete;
    YoloV4& operator=(const YoloV4&) = delete;

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
    int m_numClass = 80;
    float m_confThreshold = 0.4f;    // YOLOv4 官方用 0.4
    float m_iouThreshold = 0.6f;     // YOLOv4 官方用 0.6
    int m_topK = 300;
    float m_normScale = 255.f;       // YOLOv4: BGR->RGB + /255
    float m_normMean[3] = {0.f, 0.f, 0.f};
    float m_normStd[3]  = {1.f, 1.f, 1.f};
    float m_padValue = 114.f;

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "input"
    std::string m_outputName;   // "output"

    int m_batch = 0;
    int m_srcW = 0;
    int m_srcH = 0;

    int m_srcRow = 0;      // 4 + nc
    int m_anchors = 0;     // 22743（608 输入）

    core::CudaStream m_stream;
    core::DeviceBuffer m_inputSrc;
    core::DeviceBuffer m_resizeOut;
    core::DeviceBuffer m_inputNchw;
    core::DeviceBuffer m_outputSrc;
    core::DeviceBuffer m_objects;
    core::PinnedBuffer m_objectsHost;

    int m_objectsPerImage = 0;

    trt_alpha::kernels::AffineMat m_dst2src{};

    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
};

}  // namespace trt_alpha::det