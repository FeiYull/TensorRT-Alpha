// =============================================================================
//  trt_alpha :: core :: buffer
// -----------------------------------------------------------------------------
//  Buffer -- an owning memory container (Host or Device).
//
//  Responsibilities:
//    * allocate / free Host or Device memory
//    * record the layout information (width, height, channels, stride, type,
//      and where it lives)
//    * hand out BufferView (read-only views)
//
//  Not responsible for:
//    * colour / tensor semantics (agreed between producer and consumer)
//    * data manipulation (no copies, no conversions)
//    * deciding "whether to allocate" (the caller decides)
//
//  Design conventions:
//    * Tight allocation: stride = width * channels * sizeOf(dtype) (no padding)
//    * Copy and move are disabled: a class that owns memory is never passed by
//      value, only by shared_ptr
//    * Host memory comes from MemoryPool (Kind::Host) so the pageable block is
//      recycled instead of being handed back to the OS on every batch; Device
//      still allocates straight through cudaMalloc
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/core/memory_pool.hpp"   // MemoryPool::Kind (which free list to return to)

#include <cstddef>
#include <cstdint>
#include <memory>

namespace trt_alpha::core {

//! Owning memory container. Not copyable, not movable (passed only via shared_ptr).
class Buffer
{
public:
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    // -------------------------------------------------------------------------
    // Factories
    // -------------------------------------------------------------------------

    //! Allocate on the Host.
    //! width / height / channels must be > 0. stride = width * channels * sizeOf(dtype) (tight).
    [[nodiscard]] static std::shared_ptr<Buffer>
    createHost(int width, int height, int channels, DataType dtype);

    //! Allocate page-locked (pinned) Host memory.
    //!
    //! Why this is a separate factory rather than a flag: page-locked memory is a
    //! different resource with different costs, and only the caller knows whether
    //! it is worth it. The reason to ask for it is `cudaMemcpyAsync`: with pageable
    //! memory the driver silently turns the "async" copy into a synchronous one
    //! through its own staging buffer (measured: 6.2 MB D2H blocked the calling
    //! thread for 793 us instead of returning immediately). With page-locked memory
    //! the call really does return at once.
    //!
    //! MemorySpace stays Host -- it is host memory either way; only its
    //! pageability differs, which is a detail of which pool it came from.
    [[nodiscard]] static std::shared_ptr<Buffer>
    createPinned(int width, int height, int channels, DataType dtype);

    //! Allocate on the Device (cudaMalloc).
    //! Will switch to pooled allocation once MemoryPool lands.
    [[nodiscard]] static std::shared_ptr<Buffer>
    createDevice(int width, int height, int channels, DataType dtype);

    //! Build from Host data (row-by-row copy).
    //! srcData must point to at least srcStride * height valid bytes.
    //! srcStride is "bytes per row in the source data" (may exceed the tight value).
    [[nodiscard]] static std::shared_ptr<Buffer>
    fromHostData(const std::uint8_t* srcData, int width, int height,
                 int channels, int srcStride, DataType dtype);

    // -------------------------------------------------------------------------
    // Views
    // -------------------------------------------------------------------------

    [[nodiscard]] BufferView view() const noexcept;

    // -------------------------------------------------------------------------
    // Raw access
    // -------------------------------------------------------------------------

    [[nodiscard]] const std::uint8_t* data() const noexcept { return m_data; }
    [[nodiscard]] std::uint8_t* mutableData() noexcept { return m_data; }

    // -------------------------------------------------------------------------
    // Layout information
    // -------------------------------------------------------------------------

    [[nodiscard]] int width() const noexcept { return m_width; }
    [[nodiscard]] int height() const noexcept { return m_height; }
    [[nodiscard]] int channels() const noexcept { return m_channels; }
    [[nodiscard]] int stride() const noexcept { return m_stride; }
    [[nodiscard]] DataType dtype() const noexcept { return m_dtype; }
    [[nodiscard]] MemorySpace space() const noexcept { return m_space; }

    //! Total bytes occupied = stride * height.
    [[nodiscard]] std::size_t byteSize() const noexcept
    {
        return static_cast<std::size_t>(m_stride) * static_cast<std::size_t>(m_height);
    }

private:
    Buffer(std::uint8_t* data, int width, int height, int channels,
           int stride, DataType dtype, MemorySpace space,
           MemoryPool::Kind poolKind, std::weak_ptr<MemoryPool> pool,
           std::size_t capacity) noexcept;

    void release() noexcept;

    std::uint8_t* m_data = nullptr;
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    int m_stride = 0;
    DataType m_dtype = DataType::UInt8;
    MemorySpace m_space = MemorySpace::Host;

    //! Which pool kind the block came from. MemorySpace alone is not enough:
    //! pageable and page-locked host memory are both Host, but they must be
    //! returned to different free lists (the pool rejects a block handed back to
    //! the wrong kind).
    MemoryPool::Kind m_poolKind = MemoryPool::Kind::Host;

    std::weak_ptr<MemoryPool> m_pool;   //!< set when the block came from the pool
    std::size_t m_capacity = 0;         //!< bytes actually held (>= byteSize())
};

}  // namespace trt_alpha::core
