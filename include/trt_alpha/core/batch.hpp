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
#include "trt_alpha/core/gpu_event.hpp"

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

    //! Optional: the very same pixels, already resident in device memory.
    //!
    //! A source that decodes on the GPU (NVDEC) fills this, so the model can skip
    //! the host-to-device upload entirely -- the pixels never leave the card.
    //! `views` above still describes a host copy, because the renderer draws into
    //! it and most models can only read host memory; this field is purely an
    //! acceleration hint.
    //!
    //! nullptr means "the pixels only exist on the host" and every consumer falls
    //! back to the classic upload path. That is also what keeps a model that has
    //! not been taught about this field correct instead of crashing on a device
    //! pointer.
    //!
    //! When the source knows nobody will read the host copy (`needHostCopy` is
    //! false), `buffer` is left null and `views` describes THIS block instead --
    //! same geometry, `space == MemorySpace::Device`, `data` pointing into the
    //! device allocation. That way the frame count and layout are still
    //! discoverable, and anything that blindly dereferences `views[i].data` as a
    //! host pointer is a bug the `space` field was created to prevent.
    std::shared_ptr<Buffer> deviceBuffer;

    //! Optional: when set, `views[i].data` is only readable after wait().
    //!
    //! A source that fills the host copy with an asynchronous memcpy hands this
    //! over instead of blocking itself, so the copy overlaps with the rest of the
    //! pipeline. Readers must call wait() before touching the pixels; it is
    //! idempotent and cheap once the copy has landed.
    std::shared_ptr<GpuEvent> hostReady;

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

    //! A batch with pixels only on the device is NOT empty: `buffer` is null but
    //! `deviceBuffer` is not, and the geometry still lives in `views`.
    [[nodiscard]] bool empty() const noexcept
    {
        return views.empty() || (buffer == nullptr && deviceBuffer == nullptr);
    }

    [[nodiscard]] std::size_t size() const noexcept { return views.size(); }
};

}  // namespace trt_alpha::core
