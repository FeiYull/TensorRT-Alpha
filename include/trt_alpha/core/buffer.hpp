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
//    * Device currently allocates straight through cudaMalloc; the
//      implementation will switch to MemoryPool once it lands
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"

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
           int stride, DataType dtype, MemorySpace space) noexcept;

    void release() noexcept;

    std::uint8_t* m_data = nullptr;
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    int m_stride = 0;
    DataType m_dtype = DataType::UInt8;
    MemorySpace m_space = MemorySpace::Host;
};

}  // namespace trt_alpha::core
