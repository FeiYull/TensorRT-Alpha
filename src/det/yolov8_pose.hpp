// =============================================================================
//  trt_alpha :: det :: YoloV8Pose（私有头文件）
// -----------------------------------------------------------------------------
//  YOLOv8-pose 姿态估计。
//  - 输入 [B, 3, 640, 640]
//  - 输出 [B, 56, 8400]（4 + 1 conf + 17 kpts × 3）
//  - 需要 transpose（[B, 56, 8400] -> [B, 8400, 56]）
//  - 关键点是 (x, y, conf) 三连，网络输入坐标，postprocess 里做 m_dst2src 变换
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
#include "trt_alpha/kpt/keypointer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::det {

class YoloV8Pose final : public kpt::IKeypointer
{
public:
    YoloV8Pose() = default;
    ~YoloV8Pose() override = default;

    YoloV8Pose(const YoloV8Pose&) = delete;
    YoloV8Pose& operator=(const YoloV8Pose&) = delete;

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
    int m_numKpts = 17;          // COCO 17 关键点
    float m_confThreshold = 0.25f;
    float m_iouThreshold = 0.7f;
    int m_topK = 300;
    float m_normScale = 255.f;
    float m_normMean[3] = {0.f, 0.f, 0.f};
    float m_normStd[3]  = {1.f, 1.f, 1.f};
    float m_padValue = 114.f;

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output0"

    int m_batch = 0;
    int m_srcW = 0;
    int m_srcH = 0;

    int m_srcRow = 0;      // 5 + 17*3 = 56
    int m_anchors = 0;     // 8400
    int m_objectsRow = 0;  // 7 + 17*3 = 58

    core::CudaStream m_stream;
    core::DeviceBuffer m_inputSrc;
    core::DeviceBuffer m_resizeOut;
    core::DeviceBuffer m_inputNchw;
    core::DeviceBuffer m_outputSrc;         // [B, 56, 8400]
    core::DeviceBuffer m_outputTransposed;  // [B, 8400, 56]
    core::DeviceBuffer m_objects;           // [B, 1 + topK*58]
    core::PinnedBuffer m_objectsHost;

    int m_objectsPerImage = 0;

    trt_alpha::kernels::AffineMat m_dst2src{};

    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
};

}  // namespace trt_alpha::det