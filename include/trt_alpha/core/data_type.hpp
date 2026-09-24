// =============================================================================
//  trt_alpha :: core :: data_type
// -----------------------------------------------------------------------------
//  DataType —— 内存块中【每个元素】的类型。
//
//  为什么需要它：
//    * 同一块内存，元素类型不同，占用的字节数不同
//      （uint8 每元素 1 字节；float32 每元素 4 字节）
//    * fp16 和 bf16 都是 2 字节，但语义不同，不能混用
//    * 只有"每元素字节数"（elementSize）不够，必须知道具体类型
//
//  覆盖范围：
//    * 图像：UInt8 / UInt16 / Float32
//    * TensorRT 输入输出：Float16 / BFloat16 / Float32 / Int8 / Float8_*
//    * 通用：Bool / Int16 / UInt16 / Int32 / UInt32 / Float64
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

namespace trt_alpha::core {

enum class DataType : std::uint8_t
{
    // ---- 整型 ----
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,

    // ---- 浮点 ----
    Float16,       // IEEE 754 half
    BFloat16,      // brain float
    Float32,
    Float64,
    Float8_E4M3,   // 8-bit float，4 位指数 3 位尾数
    Float8_E5M2,   // 8-bit float，5 位指数 2 位尾数

    // ---- 布尔 ----
    Bool,
};

//! 每元素占用的字节数。
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

//! 可读名字（日志 / 调试用）。
[[nodiscard]] const char* nameOf(DataType dt) noexcept;

}  // namespace trt_alpha::core