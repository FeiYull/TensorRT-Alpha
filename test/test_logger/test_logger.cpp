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

//! 线程体（独立函数，让 __func__ 显示 "logThread" 而不是 "operator ()"）。
void logThread(int t, int lines)
{
    for (int i = 0; i < lines; ++i)
    {
        TRT_LOG_INFO("thread " << t << " line " << i);
    }
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
    // ---------------------------------------------------------------
    {
        constexpr static int kThreads = 4;
        constexpr static int kLinesPerThread = 500;
        std::atomic<int> doneCount{0};

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t)
        {
            threads.emplace_back([t, &doneCount] {
                logThread(t, kLinesPerThread);
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