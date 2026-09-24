// =============================================================================
//  trt_alpha :: core :: buffer_view
// -----------------------------------------------------------------------------
//  BufferView —— 一块内存的只读视图。
//
//  设计原则：
//    * 只描述【内存布局 + 元素类型 + 内存在哪】
//    * 只指向数据，不拥有数据（生命周期由 Buffer + shared_ptr 保证）
//    * 支持 Host / Device 两类内存（MemorySpace）
//
//  为什么叫 BufferView 而不是 ImageView：
//    * 图像（BGR8）是其中一种用法
//    * TensorRT 的输入输出张量（fp16 / fp32 / int8 / fp8）也用同一抽象
//    * "Image" 这个名字会误导
//
//  为什么带 DataType 而不是 elementSize：
//    * fp16 和 bf16 都是 2 字节，但语义不同
//    * int8 和 uint8 都是 1 字节，但语义不同
//    * 只有具体类型才能表达"这是量化张量还是图像"
// =============================================================================
#pragma once

#include "trt_alpha/core/data_type.hpp"

#include <cstddef>
#include <cstdint>

namespace trt_alpha::core {

//! 内存所在位置。
enum class MemorySpace : std::uint8_t
{
    Host,     //!< CPU 内存（可分页或页锁定）
    Device,   //!< GPU 显存
};

//! 内存块的只读视图。
//!
//! 字段含义：
//!   data     —— 指向数据首字节（不拥有）
//!   width    —— 宽度（元素个数）
//!   height   —— 高度（元素个数）
//!   stride   —— 一行的字节数（>= width * channels * sizeOf(dtype)，含 padding）
//!   channels —— 通道数（图像通常 1/3/4；张量可为 1）
//!   dtype    —— 每元素的类型
//!   space    —— 数据在 Host 还是 Device
//!
//! 使用约定：
//!   * 数据是紧密排列的（元素按 stride 分行，行内紧密）
//!   * 颜色语义 / 张量语义由上下游约定（比如"数据源产出 BGR8"）
//!   * 生命周期由外部保证（通常配 shared_ptr<Buffer> 一起用）
struct BufferView
{
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;                     //!< 一行的字节数（含 padding）
    int channels = 0;                   //!< 通道数
    DataType dtype = DataType::UInt8;   //!< 每元素类型
    MemorySpace space = MemorySpace::Host;

    //! 是否为空视图。
    [[nodiscard]] bool empty() const noexcept
    {
        return data == nullptr || width <= 0 || height <= 0 || channels <= 0;
    }

    //! 整块数据占用的字节数（含 stride padding）。
    [[nodiscard]] std::size_t byteSize() const noexcept
    {
        return static_cast<std::size_t>(stride) * static_cast<std::size_t>(height);
    }

    //! 单行的字节数（不含 padding，理想紧凑值）。
    [[nodiscard]] std::size_t tightStride() const noexcept
    {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(channels) *
               sizeOf(dtype);
    }
};

}  // namespace trt_alpha::core