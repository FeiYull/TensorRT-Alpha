// =============================================================================
//  trt_alpha :: det :: YoloV26（私有头文件）
// -----------------------------------------------------------------------------
//  仅供 src/det/yolov26.cpp 和同目录模型使用。
//
//  与同目录其它 YOLO 模型的区别：
//    * 导出的 ONNX 已经把 make_anchors / dist2bbox / "* stride" / sigmoid 全部
//      烘焙进图里 —— output0 就是"解码后的 y"，C++ 侧只需要
//      conf 过滤 + NMS + scale_boxes。
//    * 因此后处理复用 kernels/ops 的 YOLOV26 融合算子，而不是 legacy 的
//      decodeYoloV8Head + nmsFast（那套会二次解码，且数值口径不同）。
//    * 预处理同样走 kernels/ops 的五合一融合算子，正/反变换都用
//      LetterBoxInfo（同源，不会漂移）。
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
#include "trt_alpha/kernels/ops/postprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_det.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::det {

//! YOLOV26 检测模型（one2many / end2end = false）。
class YoloV26 : public IDetector
{
public:
    YoloV26() = default;
    ~YoloV26() override = default;

    YoloV26(const YoloV26&) = delete;
    YoloV26& operator=(const YoloV26&) = delete;

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

protected:
    //! 运行时标签：日志 / 异常 / name() 都用它；YOLOV11 子类只改这个值。
    std::string m_tag = "yolov26";

private:
    // ---- 配置（从 ModelConfig 解出）----
    // 注意：以下 9 个成员 + m_detections 全部由基类 IDetector 提供，
    // 这里【绝不能】重复声明 —— 重复声明会遮蔽基类成员且静默生效。
    // core::ModelConfig m_cfg;
    // int   m_numClass      = 80;
    // float m_confThreshold = 0.25f;
    // float m_iouThreshold  = 0.45f;
    // int   m_topK          = 300;
    // float m_normScale     = 255.f;
    // float m_normMean[3]   = {0.f, 0.f, 0.f};
    // float m_normStd[3]    = {1.f, 1.f, 1.f};
    // float m_padValue      = 114.f;
    // std::vector<std::vector<Detection>> m_detections;

    // ---- 运行时状态 ----
    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output0"

    int m_batch   = 0;
    int m_srcW    = 0;
    int m_srcH    = 0;

    int m_outC    = 0;   // 输出通道 = 4 + nc + extra
    int m_anchors = 0;   // 输出 anchor 数（YOLOV26n 是 8400）
    int m_extra   = 0;   // 每 anchor 的额外通道（seg 32 / pose 51，det = 0）

    // ---- Device / Host 缓冲 ----
    core::CudaStream   m_stream;
    core::DeviceBuffer m_inputSrc;    // 原图上传（uint8 BGR HWC，batch 连续）
    core::DeviceBuffer m_inputNchw;   // 融合预处理输出（float RGB CHW，网络输入）
    core::DeviceBuffer m_outputSrc;   // 引擎原始输出（[B, 4+nc+extra, anchors]）
    core::DeviceBuffer m_dets;        // 融合后处理输出（[B, topK, 6+extra]）
    core::DeviceBuffer m_counts;      // 每图有效框数（int[B]）
    core::PinnedBuffer m_detsHost;    // 上面两个的 Host 端（pinned，D2H 用）
    core::PinnedBuffer m_countsHost;

    // ---- 算子 Plan（都不可拷贝/赋值，只能 unique_ptr 持有）----
    std::unique_ptr<kernels::ops::CvPreprocessVecPlan>            m_pre;
    std::unique_ptr<kernels::ops::CvProprocessFusedVecYoloV26Plan> m_pp;

    //! 与 m_pre 同源的正/反变换参数（scale_boxes 用）
    kernels::ops::LetterBoxInfo m_lb{};

    // ---- 辅助 ----
    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void buildPlans();
    void allocateBuffers();
    void ensurePreprocessPlan();   //!< 源图几何变化时重建
};

}  // namespace trt_alpha::det
