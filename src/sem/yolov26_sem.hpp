// =============================================================================
//  trt_alpha :: sem :: YoloV26Sem（私有头文件）
// -----------------------------------------------------------------------------
//  语义分割。和 depth 是同一个套路，只有两点不同：
//
//    * output0 是 [B,H,W] 的 【UINT8】类别图，而且是网络分辨率（实测 640x640，
//      cityscapes19 的值域 0~18）。argmax 已经烘焙进图里，所以类别之间【不能】
//      双线性插值 —— 那会平均出根本不存在的类。必须用 MaskInterp::Nearest。
//    * 因此引擎输出张量的 dtype 不是 fp32，绑 buffer 时必须用原始字节指针，
//      不能走 asFloat()。这也是整个仓库里唯一一个 u8 引擎输出。
//      （注意：TRT 不会替你检查 dtype，绑错会静默读错内存。）
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/kernels/ops/scale_mask_yolov26.hpp"
#include "trt_alpha/sem/semantic_segmentor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::sem {

//! YOLOV26 语义分割。
class YoloV26Sem final : public ISemanticSegmentor
{
public:
    YoloV26Sem() = default;
    ~YoloV26Sem() override = default;

    YoloV26Sem(const YoloV26Sem&) = delete;
    YoloV26Sem& operator=(const YoloV26Sem&) = delete;

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
    // 基类 ISemanticSegmentor 已提供，切勿重复声明：
    //   core::ModelConfig m_cfg;  float m_normScale;  float m_padValue;
    //   std::vector<SemanticMask> m_semantics;

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output0"

    int m_batch = 0;
    int m_srcW  = 0;   // 原图宽
    int m_srcH  = 0;

    int m_outH = 0;    // 类别图的空间尺寸（= 网络输入）
    int m_outW = 0;

    core::CudaStream   m_stream;
    core::DeviceBuffer m_inputSrc;    // uint8 BGR HWC 原图
    core::DeviceBuffer m_inputNchw;   // float RGB CHW
    core::DeviceBuffer m_clsNet;      // 引擎输出（UINT8，[B,outH,outW]）
    core::DeviceBuffer m_clsOrig;     // nearest 回原图（UINT8，[B,srcH,srcW]）
    core::PinnedBuffer m_clsHost;

    std::unique_ptr<kernels::ops::CvPreprocessVecPlan>              m_pre;
    std::unique_ptr<kernels::ops::CvScaleMaskPlan<unsigned char>>   m_sm;

    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
    void ensurePlans();
};

}  // namespace trt_alpha::sem
