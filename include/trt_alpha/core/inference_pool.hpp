// =============================================================================
//  trt_alpha :: core :: inference_pool
// -----------------------------------------------------------------------------
//  推理池（一个模型一个池）：N 个 worker 各持独立 context + stream。
//  遵循 TensorRT 官方并发模型：
//    * 每个 IExecutionContext 只能被一个线程使用
//    * 不同线程使用不同的 context 和 stream 进行并发推理
//    * ICudaEngine 是线程安全的，可被多个 context 共享
//
//  调用时序：
//    pool.submit(batch) -> future<BatchResult>
//    future.get() 阻塞直到该 batch 完成
//
//  错误处理：
//    * 构造失败抛 std::runtime_error
//    * submit 到已关闭的池抛 ThreadPoolStopped（同步）
//    * 队列满时 submit 阻塞（Block 策略），直到有空位
//    * worker 内部错误在 future.get() 时抛
//
//  模型创建：
//    池由调用方传入 Factory（典型：ModelRegistry::create(name)）。
//    * test / CLI 不需要 include 具体模型头文件
//    * 分类/分割/未来任务类型一视同仁
//
//  资源管理：
//    * 每个 worker 独占一个 IModel 实例
//    * IModel 的 CUDA 流归模型自己所有，池不管
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/core/model_config.hpp"

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace trt_alpha::core {

//! 提交任务到已关闭的池时同步抛出的异常。
class ThreadPoolStopped : public std::runtime_error
{
public:
    ThreadPoolStopped() : std::runtime_error("inference pool has been stopped") {}
};

//! 推理池。
class InferencePool
{
public:
    //! 模型工厂：每个 worker 调用一次，返回独立实例（各自 init）。
    //! 典型实现：[] { return ModelRegistry::instance().create("yolov8"); }
    using Factory = std::function<std::unique_ptr<IModel>()>;

    //! 构造：调用 factory 创建 workers 个独立 IModel 实例。
    //! workers == 0 时用 min(hardware_concurrency, 8)。
    //! maxQueueSize == 0 时队列无上限（不推荐）。
    //! factory 为空 / 返回 nullptr / init 失败 → 抛 std::runtime_error。
    InferencePool(const ModelConfig& cfg,
                  Factory factory,
                  std::size_t workers = 0,
                  std::size_t maxQueueSize = 16);

    ~InferencePool();

    InferencePool(const InferencePool&) = delete;
    InferencePool& operator=(const InferencePool&) = delete;
    InferencePool(InferencePool&&) = delete;
    InferencePool& operator=(InferencePool&&) = delete;

    //! 提交一个 batch（异步）。
    //! - 立即返回 future
    //! - 队列满时阻塞（Block 策略）直到有空位
    //! - 池已关闭时同步抛 ThreadPoolStopped
    [[nodiscard]] std::future<BatchResult> submit(Batch batch);

    //! 等所有已提交任务完成（不关闭池）。
    void waitIdle();

    //! 关闭：拒绝新任务，等队列跑完，join 全部 worker。幂等。
    void shutdown();

    [[nodiscard]] std::size_t size() const noexcept { return m_workers.size(); }
    [[nodiscard]] std::size_t pending() const;

private:
    struct Worker
    {
        std::unique_ptr<IModel> model;   //!< 独占 IModel 实例
        std::thread thread;              //!< 独占线程
    };

    void workerLoop(std::size_t workerIndex);
    BatchResult runBatch(std::size_t workerIndex, Batch batch);

    ModelConfig m_cfg;
    Factory m_factory;
    std::size_t m_maxQueueSize;
    std::vector<Worker> m_workers;

    std::queue<std::pair<Batch, std::promise<BatchResult>>> m_tasks;

    mutable std::mutex m_mutex;
    std::condition_variable m_cvTask;      // 任务来了
    std::condition_variable m_cvNotFull;   // 队列有空位
    std::condition_variable m_cvIdle;      // 全空闲
    std::size_t m_activeCount = 0;
    bool m_stop = false;
};

}  // namespace trt_alpha::core