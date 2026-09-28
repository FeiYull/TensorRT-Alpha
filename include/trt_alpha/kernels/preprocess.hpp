// =============================================================================
//  trt_alpha :: kernels :: preprocess
// -----------------------------------------------------------------------------
//  预处理算子（CUDA）。所有指针参数都指向【Device 显存】。
//
//  AffineMat 定义在此：它由 letterbox 几何计算产生（预处理阶段的产物），
//  被 resizeLetterbox 消费；后处理只读它做逆变换。
// =============================================================================
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_alpha::kernels {

//! BGR HWC float (Device) -> RGB HWC float (Device)，in-place。
//! 不归一化、不转 layout。用于 EfficientDet（输入 NHWC）。
void bgrToRgbHwc(cudaStream_t stream, int batch,
                 float* data, int width, int height);

//! 把 float BGR HWC (Device) 原样拷到 dst 的 (padTop, padLeft) 位置，
//! 其他区域填 padValue。不做缩放。
//! src 尺寸 = srcW × srcH，dst 尺寸 = dstW × dstH。
//! 用于 YOLO-NAS 的"先缩放到 636×636，再 pad 到 640×640"。
void copyWithPadding(cudaStream_t stream, int batch,
                     const float* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, int padTop, int padLeft);

//! 2x3 仿射矩阵（网络输入坐标 → 源图坐标）。
//! letterbox 只做等比缩放 + 平移，故 v1 == v3 == 0。
struct AffineMat
{
    float v0, v1, v2;
    float v3, v4, v5;
};

//! 融合 kernel：float BGR HWC (Device) -> 归一化 + (可选 BGR→RGB) + NCHW (Device)
//! 数值语义：out = (src / scale - mean) / std
//! swapRB = true 时 BGR→RGB（YOLOv5/v6/v7/v8 用）；
//! swapRB = false 时不转通道（YOLOX 用）。
void bgrToNchwNormalized(cudaStream_t stream, int batch,
                         const float* src, float* dst,
                         int width, int height,
                         float scale, const float mean[3], const float std_[3],
                         bool swapRB = true);

//! letterbox 双线性 resize：uint8 BGR HWC (Device) -> float BGR HWC (Device)
//! padValue 通常取 114（与 ultralytics 一致）。
//! dst2src：网络输入坐标 → 源图坐标的仿射矩阵。
void resizeLetterbox(cudaStream_t stream, int batch,
                     const std::uint8_t* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, AffineMat dst2src);

//! letterbox 双线性 resize 的 float 输入版：
//! float BGR HWC (Device) -> float BGR HWC (Device)。
//! 用于 EfficientDet（H2D 时已经转成 float）。
void resizeLetterbox(cudaStream_t stream, int batch,
                     const float* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, AffineMat dst2src);

}  // namespace trt_alpha::kernels