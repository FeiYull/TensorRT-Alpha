// =============================================================================
//  trt_alpha :: core :: pinned_buffer
// -----------------------------------------------------------------------------
//  PinnedBuffer —— 页锁定主机内存的 RAII 容器（经 MemoryPool 池化）。
//
//  页锁定内存作为 cudaMemcpyAsync 的源 / 目的，带宽显著高于普通可分页内存。
//
//  设计同 DeviceBuffer；禁移动（保留一致性——若将来需要可加 move）。
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/memory_pool.hpp"

#include <cstddef>
#include <memory>

namespace trt_alpha::core {

class PinnedBuffer
{
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t bytes) { allocate(bytes); }
    ~PinnedBuffer() { release(); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&&) = delete;
    PinnedBuffer& operator=(PinnedBuffer&&) = delete;

    [[nodiscard]] void* data() noexcept { return m_data; }
    [[nodiscard]] const void* data() const noexcept { return m_data; }
    [[nodiscard]] float* asFloat() noexcept { return static_cast<float*>(m_data); }
    [[nodiscard]] const float* asFloat() const noexcept
    {
        return static_cast<const float*>(m_data);
    }
    [[nodiscard]] std::size_t bytes() const noexcept { return m_capacity; }

    void allocate(std::size_t bytes);
    void reset() noexcept { release(); }

private:
    void release() noexcept;

    std::weak_ptr<MemoryPool> m_pool;
    void* m_data = nullptr;
    std::size_t m_capacity = 0;
};

}  // namespace trt_alpha::core