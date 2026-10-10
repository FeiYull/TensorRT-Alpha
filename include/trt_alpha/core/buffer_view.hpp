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
//!
//! Usage conventions:
//!   * Data is tightly packed (elements are split into rows by stride and are
//!     tight within a row)
//!   * Colour / tensor semantics are agreed between producer and consumer (e.g.
//!     "the data source produces BGR8")
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
