// =============================================================================
//  trt_alpha :: core :: inference_pool（实现）
// =============================================================================
#include "trt_alpha/core/inference_pool.hpp"

#include "trt_alpha/core/logger.hpp"

#include <algorithm>
#include <chrono>

namespace trt_alpha::core {
namespace {

//! workers == 0 时的默认值：hardware_concurrency，上限 8。
std::size_t defaultWorkers() noexcept
{
    std::size_t n = std::thread::hardware_concurrency();
    if (n == 0)
    {
        n = 1;
    }
    return std::min<std::size_t>(n, 8);
}

}  // namespace

InferencePool::InferencePool(const ModelConfig& cfg,
                             Factory factory,
                             std::size_t workers,
                             std::size_t maxQueueSize)
    : m_cfg(cfg)
    , m_factory(std::move(factory))
    , m_maxQueueSize(maxQueueSize)
{
    if (!m_factory)
    {
        throw std::runtime_error("[InferencePool] factory is empty");
    }
    if (workers == 0)
    {
        workers = defaultWorkers();
    }

    TRT_LOG_INFO("[InferencePool] constructing with " << workers << " worker(s), "
                << "engine=" << cfg.engine << ", batchSize=" << cfg.batchSize);

    try
    {
        // ---- 阶段 1：创建 + init 全部模型（只填数据，不启线程）----
        m_workers.reserve(workers);
        for (std::size_t i = 0; i < workers; ++i)
        {
            Worker worker;
            worker.model = m_factory();
            if (worker.model == nullptr)
            {
                throw std::runtime_error("[InferencePool] factory returned nullptr");
            }
            worker.model->init(cfg);
            m_workers.push_back(std::move(worker));
        }

        // ---- 阶段 2：全部就位后，再启动线程 ----
        for (std::size_t i = 0; i < workers; ++i)
        {
            m_workers[i].thread = std::thread([this, i] { workerLoop(i); });
            TRT_LOG_INFO("[InferencePool] worker[" << i << "] ready");
        }
    }
    catch (...)
    {
        TRT_LOG_ERROR("[InferencePool] construction failed, shutting down...");
        shutdown();
        throw;
    }

    TRT_LOG_INFO("[InferencePool] ready with " << m_workers.size() << " worker(s)");
}

InferencePool::~InferencePool()
{
    shutdown();
}

std::future<BatchResult> InferencePool::submit(Batch batch)
{
    if (batch.empty())
    {
        throw std::invalid_argument("[InferencePool] batch is empty");
    }

    std::promise<BatchResult> promise;
    auto future = promise.get_future();

    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_stop)
        {
            throw ThreadPoolStopped();
        }
        // 队列满：等有空位（Block 策略）
        if (m_maxQueueSize != 0)
        {
            m_cvNotFull.wait(lock, [this] {
                return m_stop || m_tasks.size() < m_maxQueueSize;
            });
            if (m_stop)
            {
                throw ThreadPoolStopped();
            }
        }
        m_tasks.emplace(std::move(batch), std::move(promise));
    }
    m_cvTask.notify_one();
    return future;
}

void InferencePool::workerLoop(std::size_t workerIndex)
{
    for (;;)
    {
        std::pair<Batch, std::promise<BatchResult>> task;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cvTask.wait(lock, [this] { return m_stop || !m_tasks.empty(); });

            if (m_stop && m_tasks.empty())
            {
                TRT_LOG_INFO("[InferencePool] worker[" << workerIndex << "] exiting");
                return;
            }

            task = std::move(m_tasks.front());
            m_tasks.pop();
            ++m_activeCount;

            // 队列腾出空位 → 通知阻塞的 submit
            m_cvNotFull.notify_one();
        }

        try
        {
            BatchResult result = runBatch(workerIndex, std::move(task.first));
            task.second.set_value(std::move(result));
        }
        catch (const std::exception& e)
        {
            TRT_LOG_ERROR("[InferencePool] worker[" << workerIndex
                        << "] inference failed: " << e.what());
            task.second.set_exception(std::current_exception());
        }
        catch (...)
        {
            TRT_LOG_ERROR("[InferencePool] worker[" << workerIndex
                        << "] unknown exception");
            task.second.set_exception(std::current_exception());
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            --m_activeCount;
            if (m_tasks.empty() && m_activeCount == 0)
            {
                m_cvIdle.notify_all();
            }
        }
    }
}

BatchResult InferencePool::runBatch(std::size_t workerIndex, Batch batch)
{
    const auto tStart = std::chrono::steady_clock::now();
    IModel& model = *m_workers[workerIndex].model;
    const int n = batch.validCount;

    model.setBatch(batch);
    model.preprocess();
    model.infer();
    model.postprocess();

    // 组装结果
    BatchResult result;
    result.sourceId = batch.sourceId;
    result.firstFrameIndex = batch.firstFrameIndex;
    result.buffer = std::move(batch.buffer);
    result.views = std::move(batch.views);
    result.validCount = batch.validCount;

    // 模型把结果 move 进 result（基类实现，无 dynamic_cast）
    model.commitResult(result);

    const auto tEnd = std::chrono::steady_clock::now();
    result.inferenceMs =
        std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex << "] done batch of "
                << n << " in " << result.inferenceMs << " ms");

    model.reset();
    return result;
}

void InferencePool::waitIdle()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cvIdle.wait(lock, [this] { return m_tasks.empty() && m_activeCount == 0; });
}

void InferencePool::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_stop)
        {
            return;
        }
        m_stop = true;
    }
    m_cvTask.notify_all();
    m_cvNotFull.notify_all();

    for (auto& worker : m_workers)
    {
        if (worker.thread.joinable())
        {
            worker.thread.join();
        }
    }
    m_workers.clear();

    TRT_LOG_INFO("[InferencePool] shutdown complete");
}

std::size_t InferencePool::pending() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_tasks.size();
}

}  // namespace trt_alpha::core