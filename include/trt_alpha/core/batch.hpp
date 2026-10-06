// =============================================================================
//  trt_alpha :: core :: batch
// -----------------------------------------------------------------------------
//  Batch —— 一批"内存块"（BufferView 数组 + 共享 Buffer + 有效数 + 帧号）。
//  （详细注释见之前版本，本次只更新类型名）
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::core {

struct Batch
{
    int sourceId = -1;
    std::uint64_t firstFrameIndex = 0;
    std::shared_ptr<Buffer> buffer;
    std::vector<BufferView> views;
    int validCount = 0;

    //! 每帧的"来源名主干"（不含扩展名 / 序号），长度 == validCount。
    //!   * 图片源：原文件名（bus.jpg → "bus"）—— 存盘时按原文件名落盘
    //!   * 视频源：源文件主干（demo.mp4 → "demo"），同批多帧同名，序号由渲染层补
    //!   * 摄像头：cam<id>
    //! 数据源不填时为空，渲染层回退为 frame_<帧号>。
    std::vector<std::string> frameNames;

    [[nodiscard]] bool validate(std::string* errorMsg = nullptr) const;

    [[nodiscard]] bool empty() const noexcept
    {
        return buffer == nullptr || views.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept { return views.size(); }
};

}  // namespace trt_alpha::core