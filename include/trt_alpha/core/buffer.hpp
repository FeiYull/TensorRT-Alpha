// =============================================================================
//  trt_alpha :: core :: buffer
// -----------------------------------------------------------------------------
//  Buffer —— 拥有型内存容器（Host 或 Device）。
//
//  职责：
//    * 分配 / 释放 Host 或 Device 内存
//    * 记录内存的布局信息（宽、高、通道、步长、类型、在哪）
//    * 提供 BufferView（只读视图）
//
//  不负责：
//    * 颜色 / 张量语义（由上下游约定）
//    * 数据操作（不做拷贝、不做转换）
//    * "该不该分配"（调用方决定）
//
//  设计约定：
//    * 紧凑分配：stride = width * channels * sizeOf(dtype)（无 padding）
//    * 禁拷贝、禁移动：拥有内存的类不通过值传递，只通过 shared_ptr 引用
//    * Device 当前用 cudaMalloc 直接分配；等 MemoryPool 完成后替换实现
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace trt_alpha::core {

//! 拥有型内存容器。不可拷贝、不可移动（只通过 shared_ptr 传递）。
class Buffer
{
public:
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    // -------------------------------------------------------------------------
    // 工厂
    // -------------------------------------------------------------------------

    //! 在 Host 分配。
    //! 宽/高/通道必须 > 0。stride = width * channels * sizeOf(dtype)（紧凑）。
    [[nodiscard]] static std::shared_ptr<Buffer>
    createHost(int width, int height, int channels, DataType dtype);

    //! 在 Device 分配（cudaMalloc）。
    //! 等 MemoryPool 完成后改为经池分配。
    [[nodiscard]] static std::shared_ptr<Buffer>
    createDevice(int width, int height, int channels, DataType dtype);

    //! 从 Host 数据构造（逐行拷贝）。
    //! srcData 必须指向至少 srcStride * height 字节的有效内存。
    //! srcStride 是"源数据每行的字节数"（可能 > 紧凑值）。
    [[nodiscard]] static std::shared_ptr<Buffer>
    fromHostData(const std::uint8_t* srcData, int width, int height,
                 int channels, int srcStride, DataType dtype);

    // -------------------------------------------------------------------------
    // 视图
    // -------------------------------------------------------------------------

    [[nodiscard]] BufferView view() const noexcept;

    // -------------------------------------------------------------------------
    // 原始访问
    // -------------------------------------------------------------------------

    [[nodiscard]] const std::uint8_t* data() const noexcept { return m_data; }
    [[nodiscard]] std::uint8_t* mutableData() noexcept { return m_data; }

    // -------------------------------------------------------------------------
    // 布局信息
    // -------------------------------------------------------------------------

    [[nodiscard]] int width() const noexcept { return m_width; }
    [[nodiscard]] int height() const noexcept { return m_height; }
    [[nodiscard]] int channels() const noexcept { return m_channels; }
    [[nodiscard]] int stride() const noexcept { return m_stride; }
    [[nodiscard]] DataType dtype() const noexcept { return m_dtype; }
    [[nodiscard]] MemorySpace space() const noexcept { return m_space; }

    //! 总占用字节数 = stride * height。
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