// =============================================================================
//  trt_alpha :: core :: batch_result
// -----------------------------------------------------------------------------
//  BatchResult —— 一批图像的推理结果。
//
//  字段说明：
//    * sourceId / firstFrameIndex：跟 Batch 对齐（多源分辨 + 帧号）
//    * buffer / views / validCount：原图（推理时输入的那批图）
//    * detections / segmentations / classifications：各任务结果
//      （按模型类型填，未命中的任务字段保持空）
//    * inferenceMs / submitTime：性能元信息
//
//  生命周期：
//    * buffer 是 shared_ptr（保证 views[i].data 有效）
//    * 各任务结果自拥有（vector 里的 struct 值语义）
//    * 渲染阶段用 views[i] 拿原图，detections[i] 等拿结果
// =============================================================================
#pragma once

#include "trt_alpha/cls/types.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/det/types.hpp"
#include "trt_alpha/seg/types.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace trt_alpha::core {

//! 一批图像的推理结果。
struct BatchResult
{
    // ---- 源信息（跟 Batch 对齐）----
    int sourceId = -1;                           //!< 源标识
    std::uint64_t firstFrameIndex = 0;           //!< 本批首帧的帧号

    // ---- 原图（推理时的输入）----
    std::shared_ptr<Buffer> buffer;              //!< 原图所有者（一整块连续内存）
    std::vector<BufferView> views;               //!< 每张图的视图
    int validCount = 0;                          //!< 有效帧数（<= views.size()）

    // ---- 各任务结果（按模型类型填，未命中的保持空）----
    std::vector<std::vector<det::Detection>> detections;          //!< 每张图的检测结果
    std::vector<std::vector<seg::Segmentation>> segmentations;    //!< 每张图的分割结果
    std::vector<std::vector<cls::ClassScore>> classifications;    //!< 每张图的分类结果

    // ---- 性能元信息 ----
    double inferenceMs = 0.0;                    //!< 本批推理耗时（ms）
    std::chrono::steady_clock::time_point submitTime;   //!< 提交时间戳

    //! 是否为空。
    [[nodiscard]] bool empty() const noexcept
    {
        return buffer == nullptr || views.empty();
    }

    //! batch size。
    [[nodiscard]] std::size_t size() const noexcept { return views.size(); }
};

}  // namespace trt_alpha::core