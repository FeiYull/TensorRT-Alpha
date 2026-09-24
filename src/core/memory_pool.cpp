// =============================================================================
//  trt_alpha :: core :: memory_pool（实现）
// =============================================================================
#include "trt_alpha/core/memory_pool.hpp"

#include "trt_alpha/core/logger.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace trt_alpha::core {
namespace {

void checkCuda(cudaError_t err, const char* op, std::size_t bytes)
{
    if (err != cudaSuccess)
    {
        const std::string msg = std::string("MemoryPool: ") + op + " failed for " +
                                std::to_string(bytes) + " bytes: " +
                                cudaGetErrorString(err);
        TRT_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }
}

}  // namespace

std::size_t MemoryPool::roundUp(std::size_t bytes) noexcept
{
    // 256 字节粒度：与 cudaMalloc 返回的对齐保证一致，杜绝碎片键
    constexpr std::size_t kGranularity = 256;
    return ((bytes + kGranularity - 1) / kGranularity) * kGranularity;
}

std::shared_ptr<MemoryPool> MemoryPool::instance()
{
    // 局部静态 + shared_ptr：
    //   * C++11 起局部静态初始化线程安全
    //   * 进程退出时按静态析构顺序销毁；缓冲区持 weak_ptr，
    //     池先亡则它们安全退化为直接 cudaFree / cudaFreeHost
    static const std::shared_ptr<MemoryPool> pool = std::make_shared<MemoryPool>();
    return pool;
}

MemoryPool::MemoryPool(std::size_t deviceCacheLimitBytes,
                       std::size_t pinnedCacheLimitBytes)
{
    state(Kind::Device).cacheLimit = deviceCacheLimitBytes;
    state(Kind::PinnedHost).cacheLimit = pinnedCacheLimitBytes;
}

MemoryPool::~MemoryPool()
{
    for (int k = 0; k < 2; ++k)
    {
        KindState& st = m_state[k];
        std::lock_guard<std::mutex> lock(st.mutex);
        for (auto& entry : st.freeBlocks)
        {
            for (void* ptr : entry.second)
            {
                if (k == static_cast<int>(Kind::Device))
                {
                    cudaFree(ptr);
                }
                else
                {
                    cudaFreeHost(ptr);
                }
            }
        }
        st.freeBlocks.clear();
        st.stats.cachedBytes = 0;
        if (st.stats.inUseBytes > 0)
        {
            TRT_LOG_INFO("MemoryPool[" << kindName(static_cast<Kind>(k))
                        << "]: destroyed with " << st.stats.inUseBytes
                        << " byte(s) still in use by RAII owners");
        }
    }
}

MemoryPool::Block MemoryPool::allocate(Kind kind, std::size_t bytes)
{
    if (bytes == 0)
    {
        return {};
    }
    const std::size_t capacity = roundUp(bytes);
    KindState& st = state(kind);

    // ---- 1) 锁内查空闲缓存（命中路径零 CUDA 调用）----
    {
        std::lock_guard<std::mutex> lock(st.mutex);
        ++st.stats.requests;
        auto it = st.freeBlocks.lower_bound(capacity);
        if (it != st.freeBlocks.end())
        {
            void* ptr = it->second.back();
            const std::size_t cap = it->first;   // erase 前取出
            it->second.pop_back();
            if (it->second.empty())
            {
                st.freeBlocks.erase(it);
            }
            ++st.stats.hits;
            st.stats.cachedBytes -= cap;
            st.stats.inUseBytes += cap;
            st.stats.peakInUseBytes = std::max(st.stats.peakInUseBytes,
                                               st.stats.inUseBytes);
#ifndef NDEBUG
            st.outstanding.insert(ptr);
#endif
            TRT_LOG_DEBUG("MemoryPool[" << kindName(kind) << "]: allocate "
                          << bytes << " bytes -> hit (cap=" << cap << ")");
            return Block{ptr, cap};
        }
    }

    // ---- 2) 未命中：锁外做重量级 CUDA 分配（不阻塞其它线程命中）----
    void* ptr = nullptr;
    if (kind == Kind::Device)
    {
        checkCuda(cudaMalloc(&ptr, capacity), "cudaMalloc", capacity);
    }
    else
    {
        checkCuda(cudaMallocHost(&ptr, capacity), "cudaMallocHost", capacity);
    }

    // ---- 3) 锁内记账 ----
    {
        std::lock_guard<std::mutex> lock(st.mutex);
        ++st.stats.cudaAllocs;
        st.stats.inUseBytes += capacity;
        st.stats.peakInUseBytes = std::max(st.stats.peakInUseBytes, st.stats.inUseBytes);
#ifndef NDEBUG
        st.outstanding.insert(ptr);
#endif
    }
    TRT_LOG_DEBUG("MemoryPool[" << kindName(kind) << "]: allocate "
                  << bytes << " bytes -> miss (cuda alloc, cap=" << capacity << ")");
    return Block{ptr, capacity};
}

void MemoryPool::release(Kind kind, Block block) noexcept
{
    if (block.ptr == nullptr || block.capacity == 0)
    {
        return;
    }
    KindState& st = state(kind);
    bool realFree = false;
    {
        std::lock_guard<std::mutex> lock(st.mutex);
#ifndef NDEBUG
        if (st.outstanding.erase(block.ptr) == 0)
        {
            // 双重归还 / 非法指针：拒绝并打日志，绝不能让同一地址
            // 进两次空闲链（否则两个调用方会拿到同一块显存 = 数据踩踏）
            TRT_LOG_ERROR("MemoryPool[" << kindName(kind)
                          << "]: double/invalid release of pointer "
                          << block.ptr << " rejected");
            return;
        }
#endif
        st.stats.inUseBytes -= std::min(st.stats.inUseBytes, block.capacity);
        if (st.stats.cachedBytes + block.capacity <= st.cacheLimit)
        {
            st.freeBlocks[block.capacity].push_back(block.ptr);
            st.stats.cachedBytes += block.capacity;
        }
        else
        {
            realFree = true;   // 超缓存上限：锁外真释放
        }
    }
    if (realFree)
    {
        if (kind == Kind::Device)
        {
            cudaFree(block.ptr);
        }
        else
        {
            cudaFreeHost(block.ptr);
        }
        TRT_LOG_DEBUG("MemoryPool[" << kindName(kind)
                      << "]: release " << block.capacity
                      << " bytes -> real free (cache limit reached)");
    }
    else
    {
        TRT_LOG_DEBUG("MemoryPool[" << kindName(kind)
                      << "]: release " << block.capacity << " bytes -> cached");
    }
}

void MemoryPool::releaseUnused(Kind kind)
{
    KindState& st = state(kind);
    std::vector<void*> toFree;
    {
        std::lock_guard<std::mutex> lock(st.mutex);
        for (auto& entry : st.freeBlocks)
        {
            for (void* ptr : entry.second)
            {
                toFree.push_back(ptr);
            }
        }
        st.freeBlocks.clear();
        st.stats.cachedBytes = 0;
    }
    // 真释放放锁外：避免持锁执行重量级 CUDA 调用
    for (void* ptr : toFree)
    {
        if (kind == Kind::Device)
        {
            cudaFree(ptr);
        }
        else
        {
            cudaFreeHost(ptr);
        }
    }
    TRT_LOG_INFO("MemoryPool[" << kindName(kind) << "]: releaseUnused freed "
                << toFree.size() << " block(s)");
}

MemoryPool::Stats MemoryPool::stats(Kind kind) const
{
    const KindState& st = state(kind);
    std::lock_guard<std::mutex> lock(st.mutex);
    return st.stats;
}

void MemoryPool::setCacheLimit(Kind kind, std::size_t bytes)
{
    KindState& st = state(kind);
    std::lock_guard<std::mutex> lock(st.mutex);
    st.cacheLimit = bytes;
}

const char* MemoryPool::kindName(Kind kind) noexcept
{
    return kind == Kind::Device ? "device" : "pinned-host";
}

}  // namespace trt_alpha::core