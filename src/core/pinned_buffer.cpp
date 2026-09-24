// =============================================================================
//  trt_alpha :: core :: pinned_buffer（实现）
// =============================================================================
#include "trt_alpha/core/pinned_buffer.hpp"

#include "trt_alpha/core/logger.hpp"

#include <cuda_runtime.h>

namespace trt_alpha::core {

void PinnedBuffer::allocate(std::size_t bytes)
{
    if (bytes <= m_capacity)
    {
        return;
    }
    release();
    std::shared_ptr<MemoryPool> pool = MemoryPool::instance();
    const MemoryPool::Block block =
        pool->allocate(MemoryPool::Kind::PinnedHost, bytes);
    m_pool = pool;
    m_data = block.ptr;
    m_capacity = block.capacity;
}

void PinnedBuffer::release() noexcept
{
    if (m_data == nullptr)
    {
        return;
    }
    if (std::shared_ptr<MemoryPool> pool = m_pool.lock())
    {
        pool->release(MemoryPool::Kind::PinnedHost,
                      MemoryPool::Block{m_data, m_capacity});
    }
    else
    {
        cudaFreeHost(m_data);
    }
    m_data = nullptr;
    m_capacity = 0;
}

}  // namespace trt_alpha::core