// =============================================================================
//  trt_alpha :: det :: YuNet（私有头文件）
// -----------------------------------------------------------------------------
//  YuNet（libfacedetection）人脸检测 + 5 关键点。
//  - 输入 NCHW [B, 3, H, W]，H/W 动态（不 resize，直接喂原图）
//  - 3 个输出：loc [B, N, 14] / conf [B, N, 2] / iou [B, N, 1]
//  - N（候选框数）依赖输入 H/W
//  - priorBoxes 依赖输入 H/W，Host 算好上传
//  - 输出每行 17 个 float：[bbox(4) + conf + label + keep] + 5 关键点(10)
// =============================================================================
#pragma once

#include "trt_alpha/kernels/cast.hpp"
#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/det/detector.hpp"
#include "trt_alpha/kernels/postprocess.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::det {

class YuNet final : public IDetector
{
public:
    YuNet() = default;
    ~YuNet() override = default;

    YuNet(const YuNet&) = delete;
    YuNet& operator=(const YuNet&) = delete;

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
    // int m_numClass = 2;              // 背景 + 人脸
    // float m_confThreshold = 0.3f;
    // float m_iouThreshold = 0.45f;
    // int m_topK = 1000;               // 动态尺寸时候选框数可能很大

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;         // "input"
    std::string m_locName;           // "loc"
    std::string m_confName;          // "conf"
    std::string m_iouName;           // "iou"

    int m_batch = 0;
    int m_srcW = 0;
    int m_srcH = 0;

    int m_numCandidates = 0;         // N，每次 setBatch 重算（以引擎声明为准）
    int m_objectsRow = kernels::kYuNetObjectsRow;  // 7 + 10（5 关键点）

    // 常量（Device）
    core::DeviceBuffer m_minSizes;   // [4*3] = [10,16,24,32,48,FLT_MAX,64,96,FLT_MAX,128,192,256]
    core::DeviceBuffer m_variances;  // [2] = [0.1, 0.2]
    core::DeviceBuffer m_featHw;     // [4*3]（P1..P4 的 h/w/c）
    core::DeviceBuffer m_priorBoxes; // [N, 4]（每次 setBatch 重算上传）

    core::CudaStream m_stream;

    core::DeviceBuffer m_inputHwc;   // [B, 3, H, W]（float，喂 engine）
    core::DeviceBuffer m_inputU8;   // H2D 临时
    core::DeviceBuffer m_inputNchw;  // [B, 3, H, W]（HWc->CHW 后，喂 engine）
    core::DeviceBuffer m_outputLoc;  // [B, N, 14]
    core::DeviceBuffer m_outputConf; // [B, N, 2]
    core::DeviceBuffer m_outputIou;  // [B, N, 1]
    core::DeviceBuffer m_objects;    // [B, 1 + N*17]
    core::PinnedBuffer m_objectsHost;

    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateConstBuffers();
    void rebuildForSize(int W, int H);
};

}  // namespace trt_alpha::det