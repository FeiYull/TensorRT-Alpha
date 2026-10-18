// =============================================================================
//  trt_alpha :: obb :: YoloV26Obb（私有头文件）
// -----------------------------------------------------------------------------
//  YOLO26-OBB 旋转框检测。
//
//  I/O（onnxruntime 实测，见 D:\Data\trt_exp\dump_obb_ref.py）：
//      输入  images  [B, 3, 640, 640]  fp32，0~1 的 RGB
//      输出  output0 [B, 20, 8400]     = 4 (xywh 网络像素) + 15 (类别分，**已
//                                        sigmoid**) + 1 (角度，弧度)
//
//  解码（dist2rbox / ×stride / sigmoid）已经烘焙在图里，所以 C++ 只剩
//  「conf 过滤 + **旋转** NMS + 坐标还原」。
//
//  ⚠️ 旋转分支和普通检测分支**结构上不同**，不能共用融合后处理：
//      * 抑制规则是 Fast-NMS（`(iou.triu(1) >= thresh).sum(0) <= 0`），不是
//        贪心循环 —— 两者保留下来的集合不一样，不只是数值差异；
//      * IoU 是 probiou（两个高斯分布的相似度），并且类别偏移加在**中心点**
//        上（不是加在框的四条边上）。
//      实现见 kernels/ops/postprocess_fused_yolov26_obb.cu。
//
//  候选行布局（== ultralytics rotated=True 的 NMS 返回行）：
//      [cx, cy, w, h, conf, cls, angle]
//  旋转分支**不做** xywh->xyxy，所以这里保持 xywh；坐标还原走
//  ops.scale_boxes(..., xywh=True) 的口径：中心各自减 pad 再除 gain，
//  w/h 只除 gain（**不裁剪**，旋转框没有"裁进图内"这回事）。
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/kernels/ops/postprocess_fused_yolov26_obb.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/obb/obb_detector.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::obb {

class YoloV26Obb : public IObbDetector
{
public:
    YoloV26Obb() = default;
    ~YoloV26Obb() override = default;

    YoloV26Obb(const YoloV26Obb&)            = delete;
    YoloV26Obb& operator=(const YoloV26Obb&) = delete;

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
    //! 运行时标签：日志 / 异常 / name() 都用它；YOLO11 子类只改这个值。
    std::string m_tag = "yolov26_obb";

private:
    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void buildPlans();
    void allocateBuffers();
    void ensurePreprocessPlan();

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output0"

    int m_batch = 0;
    int m_srcW  = 0;
    int m_srcH  = 0;

    int m_outC    = 0;          // 4 + numClass + 1 (angle)
    int m_anchors = 0;          // 8400

    core::CudaStream   m_stream;
    core::DeviceBuffer m_inputSrc;      // u8  BGR HWC source
    core::DeviceBuffer m_inputNchw;     // f32 NCHW letterboxed
    core::DeviceBuffer m_outputSrc;     // f32 [B, outC, anchors]
    core::DeviceBuffer m_dets;          // f32 [B, topK, 7]
    core::DeviceBuffer m_counts;        // i32 [B]
    core::PinnedBuffer m_detsHost;
    core::PinnedBuffer m_countsHost;

    //! Preprocess plan + its letterbox geometry -- the inverse transform comes
    //! from the *same* object as the forward one, so the two cannot drift.
    std::unique_ptr<kernels::ops::CvPreprocessVecPlan> m_pre;
    kernels::ops::LetterBoxInfo m_lb;

    //! Move-only, so it has to live behind a unique_ptr.
    std::unique_ptr<kernels::ops::CvPostprocessFusedYoloV26ObbPlan> m_pp;
};

}  // namespace trt_alpha::obb
