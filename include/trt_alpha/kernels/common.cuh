// =============================================================================
//  trt_alpha :: kernels :: common
// -----------------------------------------------------------------------------
//  Constants and helpers shared by every kernel.
//
//  Block categories:
//    * 2D block -- pixels / anchors / decode / nms (one dimension each for dx, dy)
//    * 1D block -- reduce / cast (dx only)
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels {

//! 2D block (used by pixel / anchor kernels).
constexpr int kBlock2DX = 8;
constexpr int kBlock2DY = 8;

//! 1D block (used by reduce / cast kernels).
constexpr int kBlock1D = 256;

//! Convenience: build a 2D block.
inline dim3 block2D() noexcept
{
    return dim3(static_cast<unsigned>(kBlock2DX),
                static_cast<unsigned>(kBlock2DY));
}

//! Convenience: compute a 1D grid (rounding up).
inline int gridSize1D(std::size_t n, int block) noexcept
{
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) /
                            static_cast<std::size_t>(block));
}

//! Convenience: compute a 2D grid (x covers n, y covers batch).
inline dim3 gridSize2D(std::size_t n, int batch) noexcept
{
    return dim3(static_cast<unsigned>(gridSize1D(n, kBlock2DX)),
                static_cast<unsigned>(gridSize1D(static_cast<std::size_t>(batch),
                                                 kBlock2DY)));
}

}  // namespace trt_alpha::kernels
