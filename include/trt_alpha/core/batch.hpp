// =============================================================================
//  trt_alpha :: core :: batch
// -----------------------------------------------------------------------------
//  Batch -- a batch of "memory blocks" (an array of BufferView plus a shared
//  Buffer, a valid count and frame indices).
//  (See earlier revisions for the full commentary; this pass only renamed types.)
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

    //! Per-frame "source stem" (extension / index stripped); length == validCount.
    //!   * Image source: the original filename (bus.jpg -> "bus") -- the file is
    //!     saved under that very name.
    //!   * Video source: the source file stem (demo.mp4 -> "demo"); frames of the
    //!     same batch share it and the renderer appends the index.
    //!   * Camera: cam<id>
    //! Empty when the data source does not supply it; the renderer then falls
    //! back to frame_<index>.
    std::vector<std::string> frameNames;

    [[nodiscard]] bool validate(std::string* errorMsg = nullptr) const;

    [[nodiscard]] bool empty() const noexcept
    {
        return buffer == nullptr || views.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept { return views.size(); }
};

}  // namespace trt_alpha::core
