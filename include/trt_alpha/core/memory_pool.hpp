// =============================================================================
//  trt_alpha :: core :: memory_pool
// -----------------------------------------------------------------------------
//  MemoryPool —— 显存 + 页锁定内存的统一池。
//
//  解决的问题：
//    * cudaMalloc / cudaFree / cudaMallocHost 是重量级调用（数十 µs 起），
//      频繁分配拖垮高吞吐场景
//    * 多模型实例 / 动态 batch / 视频流尺寸变化时反复申请释放造成碎片
//
//  【安全设计】
//    * 线程安全：每个 Kind 一把互斥锁，全部状态变更都在锁内
//    * 归还凭据：Block{ptr, capacity} 必须原样来自 allocate() 的返回值
//    * Debug 双重归还检测：非 NDEBUG 构建维护在用指针登记表，重复归还
//      / 非法指针归还会被拒绝并打日志（Release 零开销）
//    * 生命周期：instance() 返回 shared_ptr，RAII 容器持 weak_ptr；
//      进程退出时池先销毁则容器退化为直接 cudaFree，无 use-after-free
//    * 缓存上限：device / pinned 各有字节上限（默认 512 MiB / 256 MiB），
//      超限的归还直接真释放，防止池无限膨胀
//
//  【高效设计】
//    * 精确尺寸空闲链：std::map<capacity, blocks> + lower_bound
//    * 256 字节粒度对齐（与 cudaMalloc 对齐保证一致）
//    * 不做 2 次幂取整（避免最多 2x 显存浪费）
//    * 命中路径 O(log n) 且零 CUDA 调用；未命中才真正 cudaMalloc，
//      且重量级 CUDA 分配在锁外执行（不阻塞其它线程的命中路径）
//    * 全量统计：命中数 / 真实 CUDA 分配数 / 峰值在用
//
//  【不做】
//    * 不用 cudaMallocAsync：拿不到精确统计和"框图 log"所需的信息；
//      且要求 CUDA 11.2+。保留扩展点（allocateAsync）在注释里。
//    * 池和流解耦：同一块内存可在不同流用；池不管"内存在哪条流上用"。
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer_view.hpp"   // MemorySpace

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace trt_alpha::core {

class MemoryPool
{
public:
    //! 内存种类。
    enum class Kind
    {
        Device,        //!< 设备显存（cudaMalloc）
        PinnedHost,    //!< 页锁定主机内存（cudaMallocHost）
    };

    //! 从池取回的块。capacity 是实际持有字节数（>= 请求的 bytes，
    //! 向上取整到 256 的倍数）。归还时必须把 allocate() 返回的 Block
    //! 原样传回 release()。
    struct Block
    {
        void* ptr = nullptr;
        std::size_t capacity = 0;
    };

    //! 池统计。
    struct Stats
    {
        std::size_t requests = 0;        //!< allocate 调用总次数
        std::size_t hits = 0;            //!< 从空闲缓存命中次数
        std::size_t cudaAllocs = 0;      //!< 真实 CUDA 分配次数
        std::size_t inUseBytes = 0;      //!< 已分配未归还字节
        std::size_t cachedBytes = 0;     //!< 空闲缓存字节（可立即复用）
        std::size_t peakInUseBytes = 0;  //!< 峰值在用字节
    };

    //! 全局池单例。
    //! 返回 shared_ptr；调用方应存 weak_ptr 以感知池生命周期。
    static std::shared_ptr<MemoryPool> instance();

    //! 构造（一般用 instance()；直接构造便于测试）。
    explicit MemoryPool(std::size_t deviceCacheLimitBytes = 512ULL << 20,
                        std::size_t pinnedCacheLimitBytes = 256ULL << 20);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    //! 申请 >= bytes 的块。CUDA 分配失败抛异常。
    //! bytes == 0 返回空块（不触 CUDA）。
    [[nodiscard]] Block allocate(Kind kind, std::size_t bytes);

    //! 归还块。noexcept —— 在 RAII 析构里调用。
    //! Debug 构建下检测并拒绝双重归还。
    void release(Kind kind, Block block) noexcept;

    //! 立即释放该种类全部空闲缓存（显存紧张时手动调用）。
    void releaseUnused(Kind kind);

    [[nodiscard]] Stats stats(Kind kind) const;
    void setCacheLimit(Kind kind, std::size_t bytes);

    static const char* kindName(Kind kind) noexcept;

private:
    struct KindState
    {
        mutable std::mutex mutex;
        std::map<std::size_t, std::vector<void*>> freeBlocks;   // capacity -> 空闲块栈
        std::size_t cacheLimit = 0;
        Stats stats;
#ifndef NDEBUG
        std::unordered_set<void*> outstanding;   // 在用指针登记（双重归还检测）
#endif
    };

    static std::size_t roundUp(std::size_t bytes) noexcept;

    KindState& state(Kind kind) noexcept
    {
        return m_state[static_cast<int>(kind)];
    }
    const KindState& state(Kind kind) const noexcept
    {
        return m_state[static_cast<int>(kind)];
    }

    KindState m_state[2];
};

}  // namespace trt_alpha::core