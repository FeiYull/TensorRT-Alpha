// =============================================================================
//  trt_alpha :: det :: EfficientDet（私有头文件）
// -----------------------------------------------------------------------------
//  EfficientDet 检测模型。
//  - 输入 NHWC [B, H, W, 3]（float32）
//  - NMS 在 engine 内（EfficientNMS_TRT 插件），输出就是最终结果
//  - 4 个输出：num_detections(int32) / detection_boxes(fp32) /
//              detection_scores(fp32) / detection_classes(int32)
//  - box 格式 [y1, x1, y2, x2]
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
#include "trt_alpha/kernels/preprocess.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::det {

class EfficientDet final : public IDetector
{
public:
    EfficientDet() = default;
    ~EfficientDet() override = default;

    EfficientDet(const EfficientDet&) = delete;
    EfficientDet& operator=(const EfficientDet&) = delete;

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
    int m_numClass = 91;             // COCO91
    float m_confThreshold = 0.45f;
    float m_padValue = 114.f;

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;          // "input"
    std::string m_numName;            // "num_detections"
    std::string m_boxesName;          // "detection_boxes"
    std::string m_scoresName;         // "detection_scores"
    std::string m_classesName;        // "detection_classes"

    int m_batch = 0;
    int m_srcW = 0;
    int m_srcH = 0;

    int m_topK = 100;                 // engine 固定 100

    core::CudaStream m_stream;
    core::DeviceBuffer m_inputSrc;    // float32 NHWC（原图，H2D 后）
    core::DeviceBuffer m_inputRgb;    // float32 NHWC（网络输入，in-place BGR->RGB 后）

    core::DeviceBuffer m_outputNum;    // int32 [B, 1]
    core::DeviceBuffer m_outputBoxes;  // fp32  [B, 100, 4]
    core::DeviceBuffer m_outputScores; // fp32  [B, 100]
    core::DeviceBuffer m_outputClasses;// int32 [B, 100]

    core::PinnedBuffer m_hostNum;      // int32 [B]
    core::PinnedBuffer m_hostBoxes;    // fp32  [B, 100, 4]
    core::PinnedBuffer m_hostScores;   // fp32  [B, 100]
    core::PinnedBuffer m_hostClasses;  // int32 [B, 100]

    trt_alpha::kernels::AffineMat m_dst2src{};

    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
};

}  // namespace trt_alpha::det