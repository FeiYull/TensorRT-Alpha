// =============================================================================
//  trt_alpha :: core :: bounded_queue
// -----------------------------------------------------------------------------
//  BoundedQueue<T> —— 有长度上限的线程安全队列。
//
//  用途：
//    * Pipeline 的"结果队列"（源线程 → 渲染线程）
//    * 将来服务化的"请求队列"
//
//  特性：
//    * 线程安全（mutex + condition_variable）
//    * 有长度上限（Bounded）
//    * 满时策略可配：DropOldest / DropNewest / Block
//    * 支持"关闭"（close）—— 关闭后 push 失败、pop 返回 false
//
//  满时策略：
//    * DropOldest —— 丢弃最旧的元素，push 新元素（实时场景首选）
//    * DropNewest —— 丢弃新元素，push 失败（保守）
//    * Block      —— 阻塞直到有空位（批处理首选）
//
//  线程模型：
//    * 多个生产者 + 多个消费者，均安全
//    * pop 阻塞直到有元素 / 队列关闭
//
//  生命周期：
//    * 构造 = 指定 maxSize + 策略
//    * close() = 唤醒所有等待者，之后 push 失败、pop 返回 false（剩余元素仍可 pop）
//    * 析构 = 自动 close
// =============================================================================
#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

namespace trt_alpha::core {

template <typename T>
class BoundedQueue
{
public:
    //! 队列满时的策略。
    enum class FullPolicy
    {
        DropOldest,   //!< 丢最旧，push 新元素（实时场景首选）
        DropNewest,   //!< 丢新元素（push 失败）
        Block,        //!< 阻塞直到有空位（批处理首选）
    };

    //! 构造。maxSize == 0 表示"无上限"（不推荐）。
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

    //! 入队。返回：
    //!   * true  —— 成功入队（可能丢弃了最旧元素）
    //!   * false —— 队列已关闭 / DropNewest 策略下已满
    //! 若丢弃了元素，通过 droppedOut 返回（非空时）。
    bool push(T value, bool* droppedOut = nullptr)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (m_closed)
        {
            return false;
        }

        // Block 策略：等队列有空位
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

        // 满了：按策略处理
        if (m_maxSize != 0 && m_queue.size() >= m_maxSize)
        {
            if (m_policy == FullPolicy::DropNewest)
            {
                if (droppedOut != nullptr) { *droppedOut = false; }
                return false;   // 丢新元素
            }
            if (m_policy == FullPolicy::DropOldest)
            {
                m_queue.pop();   // 丢最旧
                dropped = true;
            }
        }

        m_queue.push(std::move(value));
        if (droppedOut != nullptr) { *droppedOut = dropped; }
        m_cvNotEmpty.notify_one();
        return true;
    }

    //! 出队（阻塞）。返回 false 表示队列已关闭且为空。
    bool pop(T& out)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cvNotEmpty.wait(lock, [this] {
            return m_closed || !m_queue.empty();
        });

        if (m_queue.empty())
        {
            return false;   // 关闭 + 空
        }

        out = std::move(m_queue.front());
        m_queue.pop();
        m_cvNotFull.notify_one();
        return true;
    }

    //! 非阻塞出队。返回 false 表示队列空（不管关没关）。
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

    //! 关闭：唤醒所有等待者。之后 push 失败，pop 在空时返回 false。
    //! 剩余元素仍可被 pop 出来。
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

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_cvNotEmpty;
    std::condition_variable m_cvNotFull;
    std::queue<T> m_queue;
    std::size_t m_maxSize;
    FullPolicy m_policy;
    bool m_closed = false;
};

}  // namespace trt_alpha::core