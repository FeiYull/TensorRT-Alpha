// =============================================================================
//  trt_alpha :: core :: data_type
// -----------------------------------------------------------------------------
//  DataType -- the type of *each element* in a memory block.
//
//  Why it is needed:
//    * the same block occupies a different number of bytes depending on the
//      element type (uint8 = 1 byte per element; float32 = 4)
//    * fp16 and bf16 are both 2 bytes but mean different things and must not be
//      mixed
//    * knowing only the per-element byte count is not enough; the concrete type
//      must be known
//
//  Coverage:
//    * images: UInt8 / UInt16 / Float32
//    * TensorRT inputs / outputs: Float16 / BFloat16 / Float32 / Int8 / Float8_*
//    * general: Bool / Int16 / UInt16 / Int32 / UInt32 / Float64
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

namespace trt_alpha::core {

enum class DataType : std::uint8_t
{
    // ---- Integer ----
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,

    // ---- Floating point ----
    Float16,       // IEEE 754 half
    BFloat16,      // brain float
    Float32,
    Float64,
    Float8_E4M3,   // 8-bit float, 4-bit exponent, 3-bit mantissa
    Float8_E5M2,   // 8-bit float, 5-bit exponent, 2-bit mantissa

    // ---- Boolean ----
    Bool,
};

//! Bytes occupied by each element.
[[nodiscard]] constexpr std::size_t sizeOf(DataType dt) noexcept
{
    switch (dt)
    {
    case DataType::Int8:        return 1;
    case DataType::UInt8:       return 1;
    case DataType::Int16:       return 2;
    case DataType::UInt16:      return 2;
    case DataType::Int32:       return 4;
    case DataType::UInt32:      return 4;
    case DataType::Float16:     return 2;
    case DataType::BFloat16:    return 2;
    case DataType::Float32:     return 4;
    case DataType::Float64:     return 8;
    case DataType::Float8_E4M3: return 1;
    case DataType::Float8_E5M2: return 1;
    case DataType::Bool:        return 1;
    }
    return 0;   // unreachable
}

//! Human-readable name (for logs / debugging).
[[nodiscard]] const char* nameOf(DataType dt) noexcept;

}  // namespace trt_alpha::core
