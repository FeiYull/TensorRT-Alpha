// =============================================================================
//  trt_alpha :: det :: YoloV8（私有头文件）
// -----------------------------------------------------------------------------
//  仅供 src/det/yolov8.cpp 和同目录模型使用。
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

//! YOLOv8 检测模型。
class YoloV8 final : public IDetector
{
public:
    YoloV8() = default;
    ~YoloV8() override = default;

    YoloV8(const YoloV8&) = delete;
    YoloV8& operator=(const YoloV8&) = delete;

    // ---- IModel 接口 ----
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
    // ---- 配置（从 ModelConfig 解出）----
    core::ModelConfig m_cfg;
    int m_numClass = 80;
    float m_confThreshold = 0.25f;
    float m_iouThreshold = 0.45f;
    int m_topK = 300;
    float m_normScale = 255.f;
    float m_normMean[3] = {0.f, 0.f, 0.f};
    float m_normStd[3]  = {1.f, 1.f, 1.f};
    float m_padValue = 114.f;

    // ---- 运行时状态 ----
    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output0"

    int m_batch = 0;
    int m_srcW = 0;
    int m_srcH = 0;

    int m_srcRow = 0;      // 输出通道 = 4 + nc
    int m_anchors = 0;     // 输出 anchor 数（YOLOv8 通常是 8400）

    // ---- Device / Host 缓冲 ----
    core::CudaStream m_stream;
    core::DeviceBuffer m_inputSrc;        // 原图上传（uint8 BGR HWC）
    core::PinnedBuffer m_inputStaging;    // 页锁定上传暂存
    core::DeviceBuffer m_resizeOut;       // letterbox 输出（float BGR HWC）
    core::DeviceBuffer m_inputNchw;       // NCHW 归一化后（网络输入）
    core::DeviceBuffer m_outputSrc;       // 引擎原始输出（[B, 4+nc, anchors]）
    core::DeviceBuffer m_outputTransposed;// 转置后（[B, anchors, 4+nc]）
    core::DeviceBuffer m_objects;         // decode + NMS 结果（Device）
    core::PinnedBuffer m_objectsHost;     // decode + NMS 结果（Host）

    int m_objectsPerImage = 0;   // 1 + kObjectWidth * topK

    trt_alpha::kernels::AffineMat m_dst2src{};

    // ---- 辅助 ----
    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
};

}  // namespace trt_alpha::det