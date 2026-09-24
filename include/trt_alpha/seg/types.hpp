// =============================================================================
//  trt_alpha :: seg :: types
// -----------------------------------------------------------------------------
//  Segmentation —— 一个实例分割结果（框 + 掩码）。
//
//  掩码约定：
//    * mask 是 core::BufferView，语义 = CV_8UC1（255 = 前景，0 = 背景）
//    * mask.width / height 等于框在原图上的宽高（掩码已裁剪到框内）
//    * mask.data 的生命周期由 maskOwner（shared_ptr<Buffer>）保证
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/det/types.hpp"

#include <memory>

namespace trt_alpha::seg {

//! 一个实例分割结果。
struct Segmentation
{
    det::Detection box;                        //!< 框
    core::BufferView mask;                     //!< 掩码视图（CV_8UC1 语义）
    std::shared_ptr<core::Buffer> maskOwner;   //!< 掩码所有者（保证 mask.data 有效）
};

}  // namespace trt_alpha::seg