// =============================================================================
//  trt_alpha :: kpt :: YoloV26Pose（私有头文件）
// -----------------------------------------------------------------------------
//  YOLO26-pose 姿态估计。
//
//  I/O（onnxruntime 实测，见 D:\Data\trt_exp\probe_pose_seg_io.py）：
//      输入  images [B, 3, 640, 640]  fp32，0~1 的 RGB
//      输出  output0 [B, 56, 8400]    fp32
//            = 4 (xywh 网络像素) + 1 (类别分，**已 sigmoid**) + 51 (17 keypoint × 3)
//
//  和 det 一样，**解码已经烘焙在 ONNX 图里**（dist2bbox + ×stride + sigmoid +
//  关键点解码全都做完了）→ C++ 只做 conf 过滤 + NMS + 坐标还原，所以直接复用
//  CvProprocessFusedVecYoloV26Plan（extra = numKpt*3），它输出的候选行是
//
//      [x1, y1, x2, y2, conf, cls, kx0, ky0, kc0, kx1, ky1, kc1, ...]
//
//  extra 那 51 个值原样搬过来，正好就是 (x, y, conf) 三连。剩下的只是把它们
//  从网络像素坐标按同一份 LetterBoxInfo 反变换回原图（= ultralytics
//  ops.scale_coords，只缩放 x/y，且裁进图像范围；conf 通道不动）。
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/kernels/ops/postprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/kpt/keypointer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::kpt {

class YoloV26Pose : public IKeypointer
{
public:
    YoloV26Pose() = default;
    ~YoloV26Pose() override = default;

    YoloV26Pose(const YoloV26Pose&)            = delete;
    YoloV26Pose& operator=(const YoloV26Pose&) = delete;

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
    std::string m_tag = "yolov26_pose";

private:
    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void buildPlans();
    void allocateBuffers();
    void ensurePreprocessPlan();

    //! The 51 extra channels laid out as [numKpt][kptDim]; kptDim must be 3 (x, y, conf).
    int extraChannels() const noexcept { return m_numKpt * m_kptDim; }

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output0"

    int m_batch = 0;
    int m_srcW  = 0;
    int m_srcH  = 0;

    int m_outC    = 0;          // 4 + numClass + extra
    int m_anchors = 0;          // 8400
    int m_numKpt  = 17;         // COCO 17
    int m_kptDim  = 3;          // (x, y, conf)

    core::CudaStream     m_stream;
    core::DeviceBuffer   m_inputSrc;      // u8  BGR HWC source
    core::DeviceBuffer   m_inputNchw;     // f32 NCHW letterboxed
    core::DeviceBuffer   m_outputSrc;     // f32 [B, outC, anchors]
    core::DeviceBuffer   m_dets;          // f32 [B, topK, 6+extra]
    core::DeviceBuffer   m_counts;        // i32 [B]
    core::PinnedBuffer   m_detsHost;
    core::PinnedBuffer   m_countsHost;

    //! Preprocess plan + its letterbox geometry -- the inverse transform for the
    //! boxes and the keypoints comes from the *same* object, so they cannot drift.
    std::unique_ptr<kernels::ops::CvPreprocessVecPlan> m_pre;
    kernels::ops::LetterBoxInfo m_lb;

    //! Move-only, so it has to live behind a unique_ptr.
    std::unique_ptr<kernels::ops::CvProprocessFusedVecYoloV26Plan> m_pp;
};

}  // namespace trt_alpha::kpt
