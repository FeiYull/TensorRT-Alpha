// =============================================================================
//  trt_alpha :: det :: YoloNas（私有头文件）
// -----------------------------------------------------------------------------
//  YOLO-NAS 检测模型。
//  输出 [B, 8400, 84]（3 维，无 objectness，item[0..3] 直接是 xyxy 像素坐标）。
//  预处理两步：resizeLetterbox 到 636×636 → copyWithPadding 到 640×640。
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

class YoloNas final : public IDetector
{
public:
    YoloNas() = default;
    ~YoloNas() override = default;

    YoloNas(const YoloNas&) = delete;
    YoloNas& operator=(const YoloNas&) = delete;

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
    // float m_iouThreshold = 0.7f;
    // int m_topK = 300;
    // float m_normScale = 255.f;
    // float m_normMean[3] = {0.f, 0.f, 0.f};
    // float m_normStd[3]  = {1.f, 1.f, 1.f};
    // float m_padValue = 114.f;

    // YOLO-NAS 特有：先缩放到 resizeW×resizeH，再 pad 到 dstW×dstH
    int m_resizeW = 636;
    int m_resizeH = 636;
    int m_padTop = 0;
    int m_padLeft = 0;

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output"

    int m_batch = 0;
    int m_srcW = 0;
    int m_srcH = 0;

    int m_srcRow = 0;      // 4 + nc
    int m_anchors = 0;     // 8400

    core::CudaStream m_stream;
    core::DeviceBuffer m_inputSrc;
    core::DeviceBuffer m_resizeOut;         // 636×636
    core::DeviceBuffer m_inputPad;          // 640×640（copyWithPadding 后）
    core::DeviceBuffer m_inputNchw;
    core::DeviceBuffer m_outputSrc;
    core::DeviceBuffer m_objects;
    core::PinnedBuffer m_objectsHost;

    int m_objectsPerImage = 0;

    trt_alpha::kernels::AffineMat m_dst2src{};   // 按 636×636 算

    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
};

}  // namespace trt_alpha::det