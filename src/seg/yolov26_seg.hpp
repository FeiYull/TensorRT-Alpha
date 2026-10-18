// =============================================================================
//  trt_alpha :: seg :: YoloV26Seg（私有头文件）
// -----------------------------------------------------------------------------
//  YOLO26-seg 实例分割。
//
//  I/O（onnxruntime 实测，见 D:\Data\trt_exp\dump_seg_ref.py）：
//      输入  images  [B, 3, 640, 640]  fp32，0~1 的 RGB
//      输出  output0 [B, 116, 8400]    = 4 (xywh 网络像素) + 80 (类别分，**已
//                                        sigmoid**) + 32 (mask 系数)
//            output1 [B, 32, 160, 160] = mask prototype，**1/4 网络分辨率**
//
//  和 det 一样**解码已经烘焙在 ONNX 图里**（dist2bbox + ×stride + sigmoid），
//  所以直接复用 CvProprocessFusedVecYoloV26Plan（extra = 32），它输出的候选行是
//
//      [x1, y1, x2, y2, conf, cls, c0, c1, ..., c31]
//
//  mask 那一步**按官方顺序**来（ultralytics/utils/ops.py::process_mask，
//  upsample=True 分支）：
//
//      logits = coeffs @ protos.view(32, -1)     # (n, 160, 160)  <- 本仓库的
//                                                                # CvMaskCoeffDotPlan
//      up     = F.interpolate(logits, (640,640), mode="bilinear")
//                                                # <- CvScaleMaskPlan<float>
//                                                #    （逐位复刻的 ATen bilinear）
//      mask   = up > 0                           # **先上采样、后二值化**
//      mask   = crop_mask(mask, boxes)           # x >= x1 && x < x2 && y >= y1
//                                                #          && y < y2
//                                                 （注意右/下是**严格小于**）
//      keep   = mask.amax() > 0                  # 全零的实例直接丢弃
//
//  ⚠️ 顺序陷阱：老版本 ultralytics 是「先在 160 上裁框、再 resize 到框大小」，
//     本仓库的 yolov8_seg 用的就是那个顺序。当前版本改成「先上采样到网络尺寸、
//     再裁框」（上游 #24272：先裁会把双线性的边缘糊到框外）。**本模型复刻当前的
//     顺序**，所以和 yolov8_seg 的 mask 语义不同 —— 见下。
//
//  ⚠️ mask 的坐标系：官方 Results.masks 停在**网络输入分辨率**（640x640，含
//     letterbox 灰边），并不映射回原图；渲染时才由 Annotator.masks 插值到图像
//     大小。本模型照样把它停在 640x640（uint8，0/255，框外已清零），所以
//     Segmentation::mask 的 width/height 是 **dstW/dstH**，不是源图尺寸。
//     要叠到原图上，用**同一个 LetterBoxInfo**把它裁/缩回去即可。
//
//  参考实现：ultralytics/models/yolo/segment/predict.py::construct_result
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/model_config.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/kernels/ops/mask_coeff_dot_yolov26.hpp"
#include "trt_alpha/kernels/ops/postprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/kernels/ops/scale_mask_yolov26.hpp"
#include "trt_alpha/seg/segmentor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::seg {

class YoloV26Seg : public ISegmentor
{
public:
    YoloV26Seg() = default;
    ~YoloV26Seg() override = default;

    YoloV26Seg(const YoloV26Seg&)            = delete;
    YoloV26Seg& operator=(const YoloV26Seg&) = delete;

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
    std::string m_tag = "yolov26_seg";

private:
    void loadConfig(const core::ModelConfig& cfg);
    void discoverEngineIo();
    void buildPlans();
    void allocateBuffers();
    void ensurePreprocessPlan();

    std::unique_ptr<core::TrtEngine> m_engine;
    std::string m_inputName;    // "images"
    std::string m_outputName;   // "output0"
    std::string m_protoName;    // "output1"

    int m_batch = 0;
    int m_srcW  = 0;
    int m_srcH  = 0;

    int m_outC      = 0;        // 4 + numClass + numCoeffs
    int m_anchors   = 0;        // 8400
    int m_numCoeffs = 32;       // mask coefficients == prototype channels
    int m_protoH    = 0;        // 160
    int m_protoW    = 0;        // 160

    core::CudaStream   m_stream;
    core::DeviceBuffer m_inputSrc;      // u8  BGR HWC source
    core::DeviceBuffer m_inputNchw;     // f32 NCHW letterboxed
    core::DeviceBuffer m_outputSrc;     // f32 [B, outC, anchors]
    core::DeviceBuffer m_outputProtos;  // f32 [B, numCoeffs, protoH, protoW]
    core::DeviceBuffer m_dets;          // f32 [B, topK, 6+extra]
    core::DeviceBuffer m_counts;        // i32 [B]
    core::PinnedBuffer m_detsHost;
    core::PinnedBuffer m_countsHost;

    //! Mask scratch. The dot / upsample pair still runs one detection at a time
    //! (stage 2 needs a full 640x640 workspace per instance), but the device and
    //! host copies are staged kMaskChunk instances deep, so a whole chunk costs
    //! ONE cudaMemcpyAsync batch + ONE synchronize() instead of one each per
    //! instance. 8 keeps the scratch under 14 MB while cutting 39 sync points
    //! per batch down to 5.
    static constexpr int kMaskChunk = 8;
    core::DeviceBuffer m_logits;      // [kMaskChunk] x protoH*protoW
    core::DeviceBuffer m_maskUp;      // [kMaskChunk] x dstW*dstH
    core::PinnedBuffer m_maskUpHost;  // [kMaskChunk] x dstW*dstH

    //! Emit Segmentation::maskOrig (a full dstW->srcW bilinear resize per
    //! instance, ~616 us of the 1135 us per-instance budget at 1912x1080).
    //! Off by default: the built-in renderer rescales Segmentation::mask itself
    //! and never reads maskOrig. INI: [model] mask_orig.
    bool m_emitMaskOrig = false;

    //! Preprocess plan + its letterbox geometry -- the inverse transform for the
    //! boxes comes from the *same* object as the forward one, so they cannot drift.
    std::unique_ptr<kernels::ops::CvPreprocessVecPlan> m_pre;
    kernels::ops::LetterBoxInfo m_lb;

    //! All three are move-only, so they have to live behind unique_ptr.
    std::unique_ptr<kernels::ops::CvProprocessFusedVecYoloV26Plan> m_pp;
    std::unique_ptr<kernels::ops::CvMaskCoeffDotPlan>              m_dot;
    std::unique_ptr<kernels::ops::CvScaleMaskPlan<float>>          m_scale;
};

}  // namespace trt_alpha::seg
