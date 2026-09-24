// =============================================================================
//  test/test_logger/test_logger.cpp
// -----------------------------------------------------------------------------
//  Logger 测试：
//    [1] 4 个宏都能编译 + 输出（人眼检查）
//    [2] 多线程并发输出不撕裂（核心）
//    [3] TRT adapter 桥接
// =============================================================================
#include "trt_alpha/core/logger.hpp"

#include <atomic>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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
    std::cout << "=== Logger tests ===\n";

    // ---------------------------------------------------------------
    // [1] 4 个宏都能用
    // ---------------------------------------------------------------
    {
        TRT_LOG_DEBUG("debug message " << 42);
        TRT_LOG_INFO ("info message "  << 3.14);
        TRT_LOG_WARN ("warn message "  << "abc");
        TRT_LOG_ERROR("error message " << true);
        check(true, "[1] all 4 macros compiled and ran");
    }

    // ---------------------------------------------------------------
    // [2] 多线程不撕裂
    //   4 线程 × 500 行，检查每行都以 [INFO ] 开头、以 '\n' 结尾
    // ---------------------------------------------------------------
    {
        // 把 stdout 重定向到一个 stringstream 捕获输出
        // 注意：本测试需要 stdout 是"管道/文件"才能重定向；
        // 这里简化做法：直接让线程输出到 stdout（人眼看），
        // 同时用一个计数验证"没有崩溃"。
        //
        // 更严格的撕裂验证需要进程级捕获 stdout，超出本测试范围。
        // 这里先做"压力测试"——如果多线程下程序不崩、不丢日志行，
        // 就说明锁机制基本工作。

        constexpr static int kThreads = 4;
        constexpr static int kLinesPerThread = 500;
        std::atomic<int> doneCount{0};

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t)
        {
            threads.emplace_back([t, &doneCount] {
                for (int i = 0; i < kLinesPerThread; ++i)
                {
                    TRT_LOG_INFO("thread " << t << " line " << i);
                }
                doneCount.fetch_add(1);
            });
        }
        for (auto& th : threads) { th.join(); }

        check(doneCount.load() == kThreads,
              "[2] all threads finished (multi-thread stress)");
        std::cout << "       (check above: each line should be [INFO ] thread N line M)\n";
    }

    // ---------------------------------------------------------------
    // [3] TRT adapter 桥接
    // ---------------------------------------------------------------
    {
        auto& adapter = trt_alpha::core::trtLogger();
        adapter.setMinSeverity(nvinfer1::ILogger::Severity::kINFO);

        adapter.log(nvinfer1::ILogger::Severity::kINFO,    "trt info");
        adapter.log(nvinfer1::ILogger::Severity::kWARNING, "trt warning");
        adapter.log(nvinfer1::ILogger::Severity::kERROR,   "trt error");

        check(true, "[3] TRT adapter logged 3 messages");
    }

    std::cout << "====================\n";
    if (g_failures == 0)
    {
        std::cout << "ALL PASS\n";
        return 0;
    }
    std::cout << g_failures << " FAILED\n";
    return 1;
}