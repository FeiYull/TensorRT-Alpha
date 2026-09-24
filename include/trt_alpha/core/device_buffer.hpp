// =============================================================================
//  trt_alpha :: core :: device_buffer
// -----------------------------------------------------------------------------
//  DeviceBuffer —— 设备显存的 RAII 容器（经 MemoryPool 池化）。
//
//  设计：
//    * 不可拷贝；允许移动（容器语义：指针 + 容量，移动后源置空）
//    * allocate() 只增不减：容量足够时零开销直接返回（不碰池）
//    * 池生命周期解耦：仅持 weak_ptr；池先亡则退化为直接 cudaFree
//    * 析构 noexcept：真释放路径忽略 CUDA 返回值（进程退出路径）
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer_view.hpp"   // MemorySpace
#include "trt_alpha/core/memory_pool.hpp"

#include <cstddef>
#include <memory>

namespace trt_alpha::core {

class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) { allocate(bytes); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : m_pool(std::move(other.m_pool))
        , m_data(other.m_data)
        , m_capacity(other.m_capacity)
    {
        other.m_data = nullptr;
        other.m_capacity = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            m_pool = std::move(other.m_pool);
            m_data = other.m_data;
            m_capacity = other.m_capacity;
            other.m_data = nullptr;
            other.m_capacity = 0;
        }
        return *this;
    }

    [[nodiscard]] void* data() noexcept { return m_data; }
    [[nodiscard]] const void* data() const noexcept { return m_data; }
    [[nodiscard]] float* asFloat() noexcept { return static_cast<float*>(m_data); }
    [[nodiscard]] const float* asFloat() const noexcept
    {
        return static_cast<const float*>(m_data);
    }
    [[nodiscard]] std::size_t bytes() const noexcept { return m_capacity; }

    //! 分配至少 bytes 字节。容量足够时零开销返回。
    void allocate(std::size_t bytes);

    //! 显式释放（等价于析构）。
    void reset() noexcept { release(); }

private:
    void release() noexcept;

    std::weak_ptr<MemoryPool> m_pool;
    void* m_data = nullptr;
    std::size_t m_capacity = 0;
};

}  // namespace trt_alpha::core