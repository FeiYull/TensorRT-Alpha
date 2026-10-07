// =============================================================================
//  trt_alpha :: seg :: types
// -----------------------------------------------------------------------------
//  Segmentation —— 一个分割结果。
//
//  字段说明：
//    * box      —— 检测框（可选）。
//                   label == -1 表示"无框"（如 U2Net 显著性检测）。
//    * mask     —— 掩码视图（CV_8UC1 语义，整图）。
//                   - YOLOv8-seg: 0/255 二值（255=前景，0=背景），每个实例一张图
//                   - U2Net:      0~255 显著性
//    * maskOwner —— 掩码所有者，保证 mask.data 有效。
//
//  渲染器约定：
//    * box.label >= 0 时画框；否则只叠 mask。
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/det/types.hpp"

#include <memory>

namespace trt_alpha::seg {

//! 一个分割结果。
struct Segmentation
{
    det::Detection box;                        //!< 框（label == -1 表示无框）
    core::BufferView mask;                     //!< 掩码视图（CV_8UC1 语义，整图）
    std::shared_ptr<core::Buffer> maskOwner;   //!< 掩码所有者
};

}  // namespace trt_alpha::seg