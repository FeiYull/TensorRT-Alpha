// =============================================================================
//  trt_alpha :: det :: YoloX（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOX 检测模型。
//  输出 [B, 8400, 85]（decode_in_inference 已在模型内解码，含 objectness），
//  decode 复用 kernels::decodeYoloV5Head。
//  注意：batch 由引擎 profile 决定（静态引擎固定值 / 动态引擎 [min,max]）。
//  框架在 InferencePool 里统一解析并对齐 config.batch_size；模型侧另有一道
//  容量护栏（core::requireBatchCapacity），保证 views.size() 不会超过按
//  batch_size 分配的显存 —— 两层都不可省，前者管"跑多大"，后者管"别越界"。
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

class YoloX final : public IDetector
{
public:
    YoloX() = default;
    ~YoloX() override = default;

    YoloX(const YoloX&) = delete;
    YoloX& operator=(const YoloX&) = delete;

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
    // float m_confThreshold = 0.25f;
    // float m_iouThreshold = 0.45f;
    // int m_topK = 300;
    // float m_normScale = 1.f;          // YOLOX 不除以 255
    // float m_normMean[3] = {0.f, 0.f, 0.f};
    // float m_normStd[3]  = {1.f, 1.f, 1.f};
    // float m_padValue = 114.f;

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;
    std::string m_outputName;

    int m_batch = 0;
    int m_srcW = 0;
    int m_srcH = 0;

    int m_srcRow = 0;      // 5 + nc
    int m_anchors = 0;     // 8400

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