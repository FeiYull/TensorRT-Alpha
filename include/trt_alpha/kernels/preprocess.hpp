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

//! 2x3 仿射矩阵（网络输入坐标 → 源图坐标）。
//! letterbox 只做等比缩放 + 平移，故 v1 == v3 == 0。
struct AffineMat
{
    float v0, v1, v2;
    float v3, v4, v5;
};

//! letterbox 双线性 resize：uint8 BGR HWC (Device) -> float BGR HWC (Device)
//! padValue 通常取 114（与 ultralytics 一致）。
//! dst2src：网络输入坐标 → 源图坐标的仿射矩阵。
void resizeLetterbox(cudaStream_t stream, int batch,
                     const std::uint8_t* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, AffineMat dst2src);

//! 融合 kernel：float BGR HWC (Device) -> 归一化 + BGR→RGB + NCHW (Device)
//! 数值语义：out = (src / scale - mean) / std
void bgrToNchwNormalized(cudaStream_t stream, int batch,
                         const float* src, float* dst,
                         int width, int height,
                         float scale, const float mean[3], const float std_[3]);

}  // namespace trt_alpha::kernels