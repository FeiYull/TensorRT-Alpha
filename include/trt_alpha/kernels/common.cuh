// =============================================================================
//  trt_alpha :: kernels :: common
// -----------------------------------------------------------------------------
//  所有 kernel 共享的常量 + 辅助函数。
//
//  Block 分类：
//    * 2D block —— 像素 / anchor / decode / nms（dx, dy 各一维）
//    * 1D block —— reduce / cast（只有 dx）
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels {

//! 2D block（像素 / anchor kernel 用）。
constexpr int kBlock2DX = 8;
constexpr int kBlock2DY = 8;

//! 1D block（reduce / cast kernel 用）。
constexpr int kBlock1D = 256;

//! 便捷：生成 2D block。
inline dim3 block2D() noexcept
{
    return dim3(static_cast<unsigned>(kBlock2DX),
                static_cast<unsigned>(kBlock2DY));
}

//! 便捷：计算 1D grid（向上取整）。
inline int gridSize1D(std::size_t n, int block) noexcept
{
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) /
                            static_cast<std::size_t>(block));
}

//! 便捷：计算 2D grid（x 方向覆盖 n，y 方向覆盖 batch）。
inline dim3 gridSize2D(std::size_t n, int batch) noexcept
{
    return dim3(static_cast<unsigned>(gridSize1D(n, kBlock2DX)),
                static_cast<unsigned>(gridSize1D(static_cast<std::size_t>(batch),
                                                 kBlock2DY)));
}

}  // namespace trt_alpha::kernels