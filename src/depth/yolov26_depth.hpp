// =============================================================================
//  trt_alpha :: depth :: YoloV26Depth（私有头文件）
// -----------------------------------------------------------------------------
//  仅供 src/depth/yolov26_depth.cpp 使用。
//
//  与检测任务的区别（全部由导出的 ONNX 图决定，不是我们加的）：
//    * 输入仍然是 images [B,3,640,640]，前处理和检测【完全一样】——letterbox
//      640x640 + BGR2RGB + HWC2CHW + /255，所以直接复用 CvPreprocessVecPlan。
//    * output0 是 [B,1,640,640] 的【米制深度图】，已经是网络输入分辨率：
//      log-depth 的 exp(clamp(-4,5)) 与上采样都被烘焙进图里（实测 1.33~17.09 m，
//      全部 > 0）。所以 C++ 侧【没有】第二次 interpolate。
//    * 没有 anchor / DFL / NMS —— 唯一的后处理是 ops.scale_masks：
//      按 letterbox 参数裁掉灰边，再双线性缩回原图尺寸。
//
//  letterbox 的正/反参数来自同一个 LetterBoxInfo（CvPreprocessVecPlan::info()），
//  和 scale_mask 的裁剪窗由同一个 computeLetterBox 推出 —— 不会漂移。
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/depth/depth_estimator.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/kernels/ops/scale_mask_yolov26.hpp"

#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::depth {

//! YOLOV26 单目深度估计。
class YoloV26Depth final : public IDepthEstimator
{
public:
    YoloV26Depth() = default;
    ~YoloV26Depth() override = default;

    YoloV26Depth(const YoloV26Depth&) = delete;
    YoloV26Depth& operator=(const YoloV26Depth&) = delete;

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
    // 注意：以下成员全部由基类 IDepthEstimator 提供，这里【绝不能】重复声明
    //（重复声明会遮蔽基类成员且静默生效）：
    //   core::ModelConfig m_cfg;  float m_normScale;  float m_padValue;
    //   std::vector<DepthMap> m_depths;

    // ---- 运行时状态 ----
    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output0"

    int m_batch = 0;
    int m_srcW  = 0;   // 原图宽（scale_masks 的目标宽）
    int m_srcH  = 0;

    int m_outH = 0;    // output0 的空间尺寸（= 网络输入，YOLOV26n 是 640）
    int m_outW = 0;

    // ---- Device / Host 缓冲 ----
    core::CudaStream   m_stream;
    core::DeviceBuffer m_inputSrc;    // 原图上传（uint8 BGR HWC）
    core::DeviceBuffer m_inputNchw;   // 网络输入（float RGB CHW）
    core::DeviceBuffer m_depthNet;    // 引擎输出（float，[B,1,outH,outW]）
    core::DeviceBuffer m_depthOrig;   // scale_mask 输出（float，[B,srcH,srcW]）
    core::PinnedBuffer m_depthHost;   // 上面那个的 Host 端（D2H）

    // ---- 算子 Plan ----
    std::unique_ptr<kernels::ops::CvPreprocessVecPlan> m_pre;
    std::unique_ptr<kernels::ops::CvScaleMaskPlan<float>> m_sm;

    // ---- 辅助 ----
    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void allocateBuffers();
    void ensurePlans();            //!< 源图几何变化时重建
};

}  // namespace trt_alpha::depth
