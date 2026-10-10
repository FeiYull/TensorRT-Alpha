// =============================================================================
//  trt_alpha :: core :: pinned_buffer
// -----------------------------------------------------------------------------
//  PinnedBuffer -- an RAII container for page-locked host memory (pooled
//  through MemoryPool).
//
//  As the source / destination of cudaMemcpyAsync, page-locked memory offers
//  significantly higher bandwidth than ordinary pageable memory.
//
//  Designed like DeviceBuffer; move is disabled to stay consistent (it can be
//  added later if needed).
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
