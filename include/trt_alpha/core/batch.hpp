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

    [[nodiscard]] bool validate(std::string* errorMsg = nullptr) const;

    [[nodiscard]] bool empty() const noexcept
    {
        return buffer == nullptr || views.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept { return views.size(); }
};

}  // namespace trt_alpha::core