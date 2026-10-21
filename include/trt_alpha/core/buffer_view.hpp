// =============================================================================
//  trt_alpha :: core :: buffer_view
// -----------------------------------------------------------------------------
//  BufferView -- a read-only view over a block of memory.
//
//  Design principles:
//    * Describes only [memory layout + element type + where the memory lives]
//    * Points at the data without owning it (lifetime guaranteed by Buffer +
//      shared_ptr)
//    * Supports both Host and Device memory (MemorySpace)
//
//  Why it is called BufferView rather than ImageView:
//    * an image (BGR8) is just one of its uses
//    * TensorRT input / output tensors (fp16 / fp32 / int8 / fp8) share the same
//      abstraction
//    * the name "Image" would be misleading
//
//  Why it carries DataType instead of elementSize:
//    * fp16 and bf16 are both 2 bytes but mean different things
//    * int8 and uint8 are both 1 byte but mean different things
//    * only the concrete type can express "is this a quantized tensor or an image"
// =============================================================================
#pragma once

#include "trt_alpha/core/data_type.hpp"

#include <cstddef>
#include <cstdint>

namespace trt_alpha::core {

//! Where the memory lives.
enum class MemorySpace : std::uint8_t
{
    Host,     //!< CPU memory (pageable or page-locked)
    Device,   //!< GPU device memory
};

//! Pixel layout of an image block.
//!
//! Why this exists: "the data source produces BGR8" used to be an unwritten
//! agreement between producer and consumer, which works while there is exactly
//! one producer (OpenCV). A hardware decoder hands out NV12 (two planes, two
//! pitches), so the layout has to be explicit instead of agreed.
//!
//! Unknown = the block is not an image (a tensor) or the producer did not say.
enum class PixelFormat : std::uint8_t
{
    Unknown = 0,
    BGR8,       //!< 8-bit 3-channel, OpenCV's native order
    RGB8,       //!< 8-bit 3-channel, RGB order
    GRAY8,      //!< 8-bit 1-channel
    NV12,       //!< 8-bit 4:2:0, plane 0 = Y, plane 1 = interleaved UV (see plane1)
    NV12_10LE,  //!< 10-bit 4:2:0 packed in 16-bit words
    P010,       //!< 10-bit 4:2:0, MSB-aligned
};

//! Read-only view over a block of memory.
//!
//! Fields:
//!   data     -- points at the first data byte (does not own it)
//!   width    -- width in elements
//!   height   -- height in elements
//!   stride   -- bytes per row (>= width * channels * sizeOf(dtype); includes padding)
//!   channels -- number of channels (images are usually 1/3/4; a tensor may be 1)
//!   dtype    -- type of each element
//!   space    -- whether the data is on Host or Device
//!   format   -- pixel layout (BGR8 for every current source; NV12 once the
//!               hardware decoders land)
//!
//! Usage conventions:
//!   * Data is tightly packed (elements are split into rows by stride and are
//!     tight within a row)
//!   * Colour / tensor semantics come from `format`; BGR8 is the only value the
//!     current models understand
//!   * Lifetime is guaranteed externally (usually used together with
//!     shared_ptr<Buffer>)
struct BufferView
{
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;                     //!< bytes per row (includes padding)
    int channels = 0;                   //!< number of channels
    DataType dtype = DataType::UInt8;   //!< type of each element
    MemorySpace space = MemorySpace::Host;
    PixelFormat format = PixelFormat::Unknown;   //!< pixel layout of an image block

    //! Whether the view is empty.
    [[nodiscard]] bool empty() const noexcept
    {
        return data == nullptr || width <= 0 || height <= 0 || channels <= 0;
    }

    //! Bytes occupied by the whole block (including stride padding).
    [[nodiscard]] std::size_t byteSize() const noexcept
    {
        return static_cast<std::size_t>(stride) * static_cast<std::size_t>(height);
    }

    //! Bytes per row without padding (the ideal tight value).
    [[nodiscard]] std::size_t tightStride() const noexcept
    {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(channels) *
               sizeOf(dtype);
    }
};

}  // namespace trt_alpha::core
