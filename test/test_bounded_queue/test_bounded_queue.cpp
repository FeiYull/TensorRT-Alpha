// =============================================================================
//  test/test_bounded_queue/test_bounded_queue.cpp
// -----------------------------------------------------------------------------
//  BoundedQueue 测试：
//    [1] 基本 push / pop
//    [2] DropOldest：满了丢最旧
//    [3] DropNewest：满了拒绝新的
//    [4] Block：满了阻塞，直到 pop
//    [5] close：push 失败，pop 返回剩余元素后 false
//    [6] tryPop：非阻塞
// =============================================================================
#include "trt_alpha/core/bounded_queue.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;
using trt_alpha::core::BoundedQueue;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

}  // namespace

int main()
{
    std::cout << "=== BoundedQueue tests ===\n";

    // ---------------------------------------------------------------
    // [1] 基本 push / pop
    // ---------------------------------------------------------------
    {
        BoundedQueue<int> q(4);
        check(q.push(1), "[1] push(1)");
        check(q.push(2), "[1] push(2)");
        check(q.size() == 2, "[1] size == 2");

        int v = 0;
        check(q.pop(v) && v == 1, "[1] pop == 1");
        check(q.pop(v) && v == 2, "[1] pop == 2");
        check(q.size() == 0, "[1] size == 0");
    }

    // ---------------------------------------------------------------
    // [2] DropOldest
    // ---------------------------------------------------------------
    {
        BoundedQueue<int> q(2, BoundedQueue<int>::FullPolicy::DropOldest);
        q.push(1);
        q.push(2);

        bool dropped = false;
        check(q.push(3, &dropped), "[2] push(3) succeeded");
        check(dropped, "[2] dropped == true (oldest dropped)");
        check(q.size() == 2, "[2] size stays 2");

        int v = 0;
        q.pop(v);
        check(v == 2, "[2] first element is 2 (1 was dropped)");
        q.pop(v);
        check(v == 3, "[2] second element is 3");
    }

    // ---------------------------------------------------------------
    // [3] DropNewest
    // ---------------------------------------------------------------
    {
        BoundedQueue<int> q(2, BoundedQueue<int>::FullPolicy::DropNewest);
        q.push(1);
        q.push(2);
        check(!q.push(3), "[3] push(3) rejected (DropNewest)");
        check(q.size() == 2, "[3] size stays 2");

        int v = 0;
        q.pop(v);
        check(v == 1, "[3] first is 1 (unchanged)");
    }

    // ---------------------------------------------------------------
    // [4] Block
    // ---------------------------------------------------------------
    {
        BoundedQueue<int> q(1, BoundedQueue<int>::FullPolicy::Block);
        q.push(1);

        std::atomic<bool> pushed{false};
        std::thread producer([&] {
            q.push(2);   // 会被阻塞，直到消费
            pushed.store(true);
        });

        // 等一会儿，确认 producer 还没 push 成功
        std::this_thread::sleep_for(50ms);
        check(!pushed.load(), "[4] producer blocked (queue full)");

        // 消费一个，producer 应解锁
        int v = 0;
        q.pop(v);
        std::this_thread::sleep_for(50ms);
        check(pushed.load(), "[4] producer unblocked after pop");

        producer.join();
    }

    // ---------------------------------------------------------------
    // [5] close
    // ---------------------------------------------------------------
    {
        BoundedQueue<int> q(4);
        q.push(1);
        q.push(2);
        q.close();

        check(!q.push(3), "[5] push after close returns false");

        int v = 0;
        check(q.pop(v) && v == 1, "[5] pop remaining 1");
        check(q.pop(v) && v == 2, "[5] pop remaining 2");
        check(!q.pop(v), "[5] pop on closed + empty returns false");
    }

    // ---------------------------------------------------------------
    // [6] tryPop
    // ---------------------------------------------------------------
    {
        BoundedQueue<int> q(4);
        int v = 0;
        check(!q.tryPop(v), "[6] tryPop on empty returns false");

        q.push(42);
        check(q.tryPop(v) && v == 42, "[6] tryPop returns 42");
    }

    std::cout << "===========================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}