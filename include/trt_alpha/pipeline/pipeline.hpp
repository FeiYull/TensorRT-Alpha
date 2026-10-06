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
//    * 源线程内部错误（next / submit）→ 打 ERROR log，该源退出，并记入 failed()
//    * 渲染线程内部错误（推理 / 画 / 存 / 显）→ 打 ERROR log，继续，并记入 failed()
//    * 线程内的错误拿不到异常出口（join 会吞掉），因此用 failed() / firstError()
//      把"跑过但没跑成"这件事交回调用方 —— 通常用来决定进程退出码。
//      口径：**线程里任一步失败 ⇒ failed() == true**（只看有没有错，不看错在哪一步）。
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
#include <string>
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

    //! 从结果队列取一个 BatchResult（阻塞）。返回 false = 流结束。
    //! 只在没渲染线程（renderer == nullptr）时使用。
    bool popResult(core::BatchResult& out);

    //! 从"渲染后结果队列"取一个结果（阻塞）。
    //! 只在有渲染线程（renderer != nullptr）时使用。
    bool popProcessed(core::BatchResult& out);

    //! 是否起了渲染线程（renderer != nullptr）。
    [[nodiscard]] bool hasRenderer() const noexcept { return m_cfg.renderer != nullptr; }

    [[nodiscard]] bool running() const noexcept { return m_running.load(); }

    //! 本次运行是否出现过错误（源读取 / 提交、推理、画 / 存 / 显任一步失败）。
    //! 只在 waitForCompletion() 之后读取才完整。
    [[nodiscard]] bool failed() const noexcept { return m_failed.load(); }

    //! 第一条错误描述（无错误时为空串）。用于在调用方汇总成一行报出。
    [[nodiscard]] std::string firstError() const;

private:
    void sourceLoop(std::size_t sourceIndex);
    void renderLoop();

    //! 记录一条错误：置 failed 标记 + 记住首条描述（线程安全、幂等）。
    void markFailed(const std::string& what);

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

    // 错误记录：标记用 atomic（热路径无锁判断），首条描述串受 m_mutex 保护
    std::atomic<bool> m_failed{false};
    std::string m_firstError;   //!< 受 m_mutex 保护

    // 等待所有源结束 + 队列处理完
    mutable std::mutex m_mutex;
    std::condition_variable m_cvDone;
    std::size_t m_sourcesRunning = 0;
    std::unique_ptr<core::BoundedQueue<core::BatchResult>> m_processedQueue;
};

}  // namespace trt_alpha::pipeline