// =============================================================================
//  trt_alpha :: core :: bounded_queue
// -----------------------------------------------------------------------------
//  BoundedQueue<T> -- a thread-safe queue with an upper bound on its length.
//
//  Used for:
//    * Pipeline's "result queue" (source thread -> render thread)
//    * a future service-oriented "request queue"
//
//  Features:
//    * thread-safe (mutex + condition_variable)
//    * bounded length (Bounded)
//    * configurable full policy: DropOldest / DropNewest / Block
//    * supports "close" -- after which push fails and pop returns false
//
//  Full policies:
//    * DropOldest -- drop the oldest element, push the new one (preferred for real time)
//    * DropNewest -- drop the new element, push fails (conservative)
//    * Block      -- block until there is room (preferred for batch processing)
//
//  Threading model:
//    * safe with multiple producers and multiple consumers
//    * pop blocks until an element arrives / the queue is closed
//
//  Lifetime:
//    * construction = give maxSize + policy
//    * close() = wake every waiter; afterwards push fails and pop returns false
//      (remaining elements can still be popped)
//    * destruction = closes automatically
// =============================================================================
#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

namespace trt_alpha::core {

//! Policy when the queue is full.
//! It is independent of the element type, hence declared outside the template:
//! the configuration layer (pipeline::PipelineConfig) needs to hold it without
//! knowing the element type, otherwise every element type would need its own enum.
enum class QueueFullPolicy
{
    DropOldest,   //!< drop the oldest, push the new one (preferred for real time)
    DropNewest,   //!< drop the new element (push fails)
    Block,        //!< block until there is room (preferred for batch processing)
};

template <typename T>
class BoundedQueue
{
public:
    //! Compatibility alias: the old BoundedQueue<T>::FullPolicy spelling still works.
    using FullPolicy = QueueFullPolicy;

    //! Constructor. maxSize == 0 means "no upper bound" (not recommended).
    explicit BoundedQueue(std::size_t maxSize = 32,
                          FullPolicy policy = FullPolicy::DropOldest)
        : m_maxSize(maxSize)
        , m_policy(policy)
    {
    }

    ~BoundedQueue() { close(); }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;
    BoundedQueue(BoundedQueue&&) = delete;
    BoundedQueue& operator=(BoundedQueue&&) = delete;

    //! Enqueue. Returns:
    //!   * true  -- enqueued (the oldest element may have been dropped)
    //!   * false -- the queue is closed, or full under the DropNewest policy
    //! If an element was dropped, that is reported through droppedOut (when non-null).
    bool push(T value, bool* droppedOut = nullptr)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (m_closed)
        {
            return false;
        }

        // Block policy: wait for room
        if (m_policy == FullPolicy::Block && m_maxSize != 0)
        {
            m_cvNotFull.wait(lock, [this] {
                return m_closed || m_queue.size() < m_maxSize;
            });
            if (m_closed)
            {
                return false;
            }
        }

        bool dropped = false;

        // Full: apply the policy
        if (m_maxSize != 0 && m_queue.size() >= m_maxSize)
        {
            if (m_policy == FullPolicy::DropNewest)
            {
                if (droppedOut != nullptr) { *droppedOut = false; }
                return false;   // drop the new element
            }
            if (m_policy == FullPolicy::DropOldest)
            {
                m_queue.pop();   // drop the oldest
                ++m_dropped;     // leave a trace: the drop count must be reportable upstream, never silent
                dropped = true;
            }
        }

        m_queue.push(std::move(value));
        if (droppedOut != nullptr) { *droppedOut = dropped; }
        m_cvNotEmpty.notify_one();
        return true;
    }

    //! Dequeue (blocking). Returning false means the queue is closed and empty.
    bool pop(T& out)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cvNotEmpty.wait(lock, [this] {
            return m_closed || !m_queue.empty();
        });

        if (m_queue.empty())
        {
            return false;   // closed + empty
        }

        out = std::move(m_queue.front());
        m_queue.pop();
        m_cvNotFull.notify_one();
        return true;
    }

    //! Non-blocking dequeue. Returning false means the queue is empty (either way).
    bool tryPop(T& out)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty())
        {
            return false;
        }
        out = std::move(m_queue.front());
        m_queue.pop();
        m_cvNotFull.notify_one();
        return true;
    }

    //! Close: wake every waiter. Afterwards push fails and pop returns false when
    //! the queue is empty. Remaining elements can still be popped.
    void close()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_closed = true;
        }
        m_cvNotEmpty.notify_all();
        m_cvNotFull.notify_all();
    }

    [[nodiscard]] bool closed() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_closed;
    }

    [[nodiscard]] std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

    [[nodiscard]] std::size_t maxSize() const noexcept { return m_maxSize; }
    [[nodiscard]] FullPolicy policy() const noexcept { return m_policy; }

    //! Total number of dropped elements (only DropOldest produces any). Used by
    //! the caller to report "how many frames were dropped".
    [[nodiscard]] std::size_t droppedTotal() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_dropped;
    }

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_cvNotEmpty;
    std::condition_variable m_cvNotFull;
    std::queue<T> m_queue;
    std::size_t m_maxSize;
    FullPolicy m_policy;
    std::size_t m_dropped = 0;
    bool m_closed = false;
};

}  // namespace trt_alpha::core
