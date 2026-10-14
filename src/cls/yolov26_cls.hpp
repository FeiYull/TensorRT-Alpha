// =============================================================================
//  trt_alpha :: cls :: YoloV26Cls（私有头文件）
// -----------------------------------------------------------------------------
//  图像分类。六个 yolo26 任务里后处理最轻的一个：
//
//    * 预处理【不是】letterbox，是 PIL 口径的 Resize(短边 → 224) + CenterCrop
//      (224x224) + BGR2RGB + HWC2CHW + /255（见 preprocess_fused_vec_yolov26_cls）。
//      几何由 computePilClsGeometry 推出，与 torchvision 的
//      _compute_resized_output_size（长边用 int() 截断）+ center_crop
//      （int(round((image-crop)/2.0))）逐条对齐。
//      实测确认：810x1080 -> Resize 224x298（不是 299），800x657 -> 272x224
//      （不是 273），即官方用的是截断而不是四舍五入。
//    * output0 是 [B,1000]，而且 softmax【已经烘焙进图里】（实测 min>=0、
//      sum==1）。所以后处理只有一件事：top-k。
//    * 没有 NMS、没有 anchor、没有 scale_boxes，也没有 iou/conf 阈值 ——
//      所以基类里干脆不给这两个字段。
// =============================================================================
#pragma once

#include "trt_alpha/cls/classifier.hpp"
#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_cls.hpp"
#include "trt_alpha/kernels/ops/topk_yolov26_cls.hpp"

#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::cls {

//! YOLOV26 图像分类。
class YoloV26Cls final : public IClassifier
{
public:
    YoloV26Cls() = default;
    ~YoloV26Cls() override = default;

    YoloV26Cls(const YoloV26Cls&) = delete;
    YoloV26Cls& operator=(const YoloV26Cls&) = delete;

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
    // 基类 IClassifier 已提供，切勿重复声明：
    //   core::ModelConfig m_cfg;  int m_numClass;  int m_topK;
    //   float m_normScale;  float m_padValue;  std::vector<std::vector<ClassScore>> m_classifications;

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output0"

    int m_batch = 0;
    int m_srcW  = 0;
    int m_srcH  = 0;
    int m_scores = 0;           //!< output0 的长度（ImageNet = 1000）

    core::CudaStream   m_stream;
    core::DeviceBuffer m_inputSrc;    // uint8 BGR HWC 原图
    core::DeviceBuffer m_inputNchw;   // float RGB CHW（网络输入）
    core::DeviceBuffer m_logits;      // 引擎输出（float，[B, numClass]）
    core::DeviceBuffer m_topk;        // top-k 结果（float，[B, topK*2] = (label,score)）
    core::PinnedBuffer m_topkHost;

    std::unique_ptr<kernels::ops::CvPreprocessVecClsPlan> m_pre;
    std::unique_ptr<kernels::ops::CvTopKPlan>             m_topkPlan;

    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
    void ensurePlans();
};

}  // namespace trt_alpha::cls
