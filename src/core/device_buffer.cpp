// =============================================================================
//  trt_alpha :: core :: device_buffer（实现）
// =============================================================================
#include "trt_alpha/core/device_buffer.hpp"

#include "trt_alpha/core/logger.hpp"

#include <cuda_runtime.h>

namespace trt_alpha::core {

void DeviceBuffer::allocate(std::size_t bytes)
{
    if (bytes <= m_capacity)
    {
        return;   // 容量够：零开销
    }
    release();
    std::shared_ptr<MemoryPool> pool = MemoryPool::instance();
    const MemoryPool::Block block = pool->allocate(MemoryPool::Kind::Device, bytes);
    m_pool = pool;
    m_data = block.ptr;
    m_capacity = block.capacity;
}

void DeviceBuffer::release() noexcept
{
    if (m_data == nullptr)
    {
        return;
    }
    if (std::shared_ptr<MemoryPool> pool = m_pool.lock())
    {
        pool->release(MemoryPool::Kind::Device, MemoryPool::Block{m_data, m_capacity});
    }
    else
    {
        // 池已销毁（进程退出路径）：直接释放，保证无泄漏
        cudaFree(m_data);
    }
    m_data = nullptr;
    m_capacity = 0;
}

}  // namespace trt_alpha::core