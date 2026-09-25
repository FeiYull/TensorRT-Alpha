// =============================================================================
//  trt_alpha :: pipeline :: pipeline
// -----------------------------------------------------------------------------
//  Pipeline —— 三级流水线调度器。
//
//  线程模型：
//    * 数据源线程：每源一个（N 个）—— 读帧 → submit → 推结果队列
//    * 推理 worker：池内已有（M 个）—— Pipeline 不管
//    * 渲染线程：1 个 —— 从结果队列取 → 渲染 → 存 / 显
//  总计 N + M + 1 个线程。
//
//  调用时序：
//    Pipeline p(cfg);
//    p.start();               // 起 N 个源线程 + 1 个渲染线程
//    p.waitForCompletion();   // 阻塞等所有源结束 + 所有结果渲染完
//
//  停止：
//    p.stop();                // 请求停止（异步）—— 所有源 requestStop + 队列 close
//    p.waitForCompletion();   // 等线程收尾（队列里的任务依然会跑完）
//
//  错误处理：
//    * 构造 / start 失败抛异常
//    * 源线程内部错误 → 打 ERROR log，该源退出
//    * 渲染线程内部错误 → 打 ERROR log，继续
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/bounded_queue.hpp" 
#include "trt_alpha/pipeline/pipeline_config.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace trt_alpha::pipeline {

class Pipeline
{
public:
    explicit Pipeline(PipelineConfig cfg);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;

    //! 启动：起 N 个源线程 + 1 个渲染线程。失败抛异常。
    void start();

    //! 请求停止（异步，立刻返回）。线程安全。
    void stop();

    //! 阻塞等所有源结束 + 所有结果渲染完。
    void waitForCompletion();

    [[nodiscard]] bool running() const noexcept { return m_running.load(); }

private:
    void sourceLoop(std::size_t sourceIndex);
    void renderLoop();

    void validateConfig();

    PipelineConfig m_cfg;

    // 结果队列：future<BatchResult>
    std::unique_ptr<core::BoundedQueue<std::future<core::BatchResult>>> m_resultQueue;

    // 线程
    std::vector<std::thread> m_sourceThreads;
    std::thread m_renderThread;

    // 生命周期
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_stopRequested{false};

    // 等待所有源结束 + 队列处理完
    mutable std::mutex m_mutex;
    std::condition_variable m_cvDone;
    std::size_t m_sourcesRunning = 0;
};

}  // namespace trt_alpha::pipeline