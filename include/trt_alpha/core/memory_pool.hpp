// =============================================================================
//  trt_alpha :: core :: memory_pool
// -----------------------------------------------------------------------------
//  MemoryPool -- a unified pool for device memory and page-locked memory.
//
//  Problems it solves:
//    * cudaMalloc / cudaFree / cudaMallocHost are heavyweight calls (tens of us
//      and up); allocating frequently drags down high-throughput scenarios
//    * repeated allocate-free cycles caused by multiple model instances /
//      dynamic batches / changing video-stream sizes lead to fragmentation
//
//  [Safety design]
//    * Thread safety: one mutex per Kind; every state change happens under it
//    * Return credentials: the Block{ptr, capacity} must come back to release()
//      exactly as allocate() returned it
//    * Double-free detection: [every build] maintains a registry of pointers in
//      use; a duplicate or foreign pointer is rejected and logged. This guard
//      must not be compiled out in Release -- if the same pointer entered the
//      free list twice, two callers would receive the same device memory (data
//      corruption), and Release is exactly the build used for shipping and
//      benching. The cost is one hash insert/erase, which is negligible.
//    * Lifetime: instance() returns a shared_ptr and RAII containers hold a
//      weak_ptr; if the pool is destroyed first at process exit, containers
//      degrade to a direct cudaFree, so there is no use-after-free
//    * Cache limits: device / pinned each have a byte ceiling (512 MiB / 256 MiB
//      by default); returns beyond it are truly freed, so the pool cannot grow
//      without bound
//
//  [Efficiency design]
//    * Exact-size free lists: std::map<capacity, blocks> plus lower_bound
//    * 256-byte granularity alignment (matching cudaMalloc's alignment guarantee)
//    * No rounding up to powers of two (avoids up to 2x device-memory waste)
//    * The hit path is O(log n) with zero CUDA calls; only a miss really calls
//      cudaMalloc, and that heavyweight allocation runs outside the lock so it
//      does not block other threads' hit path
//    * Full statistics: hit count / real CUDA allocation count / peak in use
//
//  [Not done]
//    * cudaMallocAsync is not used: it cannot provide the exact statistics and
//      "per-block log" we need, and it requires CUDA 11.2+. The extension point
//      (allocateAsync) is kept in a comment.
//    * The pool and streams are decoupled: one block may be used on different
//      streams; the pool does not care which stream the memory is used on.
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
    //! Kinds of memory.
    enum class Kind
    {
        Device,        //!< device memory (cudaMalloc)
        PinnedHost,    //!< page-locked host memory (cudaMallocHost)
    };

    //! A block taken from the pool. capacity is the bytes actually held
    //! (>= the requested bytes, rounded up to a multiple of 256). When returning
    //! it, the Block from allocate() must be passed back to release() unchanged.
    struct Block
    {
        void* ptr = nullptr;
        std::size_t capacity = 0;
    };

    //! Pool statistics.
    struct Stats
    {
        std::size_t requests = 0;        //!< total allocate calls
        std::size_t hits = 0;            //!< hits from the free cache
        std::size_t cudaAllocs = 0;      //!< real CUDA allocations
        std::size_t inUseBytes = 0;      //!< bytes allocated but not yet returned
        std::size_t cachedBytes = 0;     //!< bytes in the free cache (immediately reusable)
        std::size_t peakInUseBytes = 0;  //!< peak bytes in use
    };

    //! Global pool singleton. Returns a const reference (a process-wide singleton
    //! living as long as the program); callers should keep a weak_ptr to observe
    //! the pool's lifetime.
    static const std::shared_ptr<MemoryPool>& instance();

    //! Constructor (normally use instance(); direct construction helps tests).
    explicit MemoryPool(std::size_t deviceCacheLimitBytes = 512ULL << 20,
                        std::size_t pinnedCacheLimitBytes = 256ULL << 20);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    //! Request a block of >= bytes. A failed CUDA allocation throws.
    //! bytes == 0 returns an empty block (no CUDA call).
    [[nodiscard]] Block allocate(Kind kind, std::size_t bytes);

    //! Return a block. noexcept -- called from RAII destructors.
    //! A double free or a foreign pointer is always rejected with an ERROR log
    //! (in every build).
    void release(Kind kind, Block block) noexcept;

    //! Free this kind's entire idle cache right away (call manually when memory
    //! is tight).
    void releaseUnused(Kind kind);

    [[nodiscard]] Stats stats(Kind kind) const;
    void setCacheLimit(Kind kind, std::size_t bytes);

    static const char* kindName(Kind kind) noexcept;

private:
    struct KindState
    {
        mutable std::mutex mutex;
        std::map<std::size_t, std::vector<void*>> freeBlocks;   // capacity -> stack of free blocks
        std::size_t cacheLimit = 0;
        Stats stats;
        std::unordered_set<void*> outstanding;   // registry of pointers in use (double-free detection)
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
