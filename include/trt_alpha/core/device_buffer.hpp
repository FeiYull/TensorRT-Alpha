// =============================================================================
//  trt_alpha :: core :: device_buffer
// -----------------------------------------------------------------------------
//  DeviceBuffer -- an RAII container for device memory (pooled through
//  MemoryPool).
//
//  Design:
//    * Not copyable; movable (container semantics: pointer + capacity; the
//      source is nulled on move)
//    * allocate() only grows: when the capacity suffices it returns immediately
//      at zero cost (without touching the pool)
//    * Decoupled from the pool's lifetime: it holds only a weak_ptr, so if the
//      pool dies first it degrades to a direct cudaFree
//    * The destructor is noexcept: the real release path ignores CUDA return
//      values (it runs on the process-exit path)
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

    //! Allocate at least `bytes`. Returns at zero cost when the capacity suffices.
    void allocate(std::size_t bytes);

    //! Explicit release (equivalent to the destructor).
    void reset() noexcept { release(); }

private:
    void release() noexcept;

    std::weak_ptr<MemoryPool> m_pool;
    void* m_data = nullptr;
    std::size_t m_capacity = 0;
};

}  // namespace trt_alpha::core
