// =============================================================================
//  trt_alpha :: core :: inference_pool（实现）
// =============================================================================
#include "trt_alpha/core/inference_pool.hpp"

#include "trt_alpha/core/logger.hpp"

#include <algorithm>
#include <atomic>
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

//! 全局任务序号（诊断用：哪个任务被哪个 worker 取走）。
std::atomic<std::uint64_t> g_taskCounter{0};

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
                << "engine=" << cfg.engine << ", batchSize=" << cfg.batchSize
                << ", maxQueueSize=" << maxQueueSize);

    try
    {
        // ---- 阶段 1：创建 + init 全部模型（只填数据，不启线程）----
        m_workers.reserve(workers);
        for (std::size_t i = 0; i < workers; ++i)
        {
            TRT_LOG_DEBUG("[InferencePool] creating model for worker[" << i << "]");
            Worker worker;
            worker.model = m_factory();
            if (worker.model == nullptr)
            {
                throw std::runtime_error("[InferencePool] factory returned nullptr");
            }
            TRT_LOG_DEBUG("[InferencePool] worker[" << i << "] calling init()");
            worker.model->init(cfg);
            TRT_LOG_DEBUG("[InferencePool] worker[" << i << "] init() done");
            m_workers.push_back(std::move(worker));
        }

        // ---- 阶段 2：全部就位后，再启动线程 ----
        for (std::size_t i = 0; i < workers; ++i)
        {
            TRT_LOG_DEBUG("[InferencePool] launching thread for worker[" << i << "]");
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

    const int batchN = batch.validCount;
    const std::uint64_t taskId = g_taskCounter.fetch_add(1);

    TRT_LOG_DEBUG("[InferencePool] submit task #" << taskId
                 << ": validCount=" << batchN << "/" << batch.views.size()
                 << ", queue=" << m_tasks.size() << "/" << m_maxQueueSize);

    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_stop)
        {
            TRT_LOG_ERROR("[InferencePool] submit rejected: pool stopped");
            throw ThreadPoolStopped();
        }
        // 队列满：等有空位（Block 策略）
        if (m_maxQueueSize != 0)
        {
            if (m_tasks.size() >= m_maxQueueSize)
            {
                TRT_LOG_WARN("[InferencePool] submit task #" << taskId
                            << " BLOCKED: queue full (" << m_tasks.size()
                            << "/" << m_maxQueueSize << ")");
            }
            m_cvNotFull.wait(lock, [this] {
                return m_stop || m_tasks.size() < m_maxQueueSize;
            });
            if (m_stop)
            {
                TRT_LOG_ERROR("[InferencePool] submit rejected after wait: pool stopped");
                throw ThreadPoolStopped();
            }
            TRT_LOG_DEBUG("[InferencePool] submit task #" << taskId
                         << " unblocked, queue=" << m_tasks.size());
        }
        m_tasks.emplace(std::move(batch), std::move(promise));
    }
    m_cvTask.notify_one();

    TRT_LOG_DEBUG("[InferencePool] submit task #" << taskId << " queued, queue="
                 << m_tasks.size());

    return future;
}

void InferencePool::workerLoop(std::size_t workerIndex)
{
    TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex << "] loop entered");

    for (;;)
    {
        std::pair<Batch, std::promise<BatchResult>> task;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex
                         << "] waiting for task, queue=" << m_tasks.size());

            m_cvTask.wait(lock, [this] { return m_stop || !m_tasks.empty(); });

            if (m_stop && m_tasks.empty())
            {
                TRT_LOG_INFO("[InferencePool] worker[" << workerIndex << "] exiting");
                return;
            }

            TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex
                         << "] picked task, queue before=" << m_tasks.size());

            task = std::move(m_tasks.front());
            m_tasks.pop();
            ++m_activeCount;

            TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex
                         << "] popped task, queue after=" << m_tasks.size()
                         << ", activeCount=" << m_activeCount);

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
            TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex
                         << "] task done, activeCount=" << m_activeCount);
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

    TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex << "] runBatch: n=" << n
                 << "/" << batch.views.size());

    const auto tPreStart = std::chrono::steady_clock::now();
    model.setBatch(batch);
    const auto tPreEnd = std::chrono::steady_clock::now();
    TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex << "] setBatch done in "
                 << std::chrono::duration<double, std::milli>(tPreEnd - tPreStart).count()
                 << " ms");

    const auto tPreProcStart = std::chrono::steady_clock::now();
    model.preprocess();
    const auto tPreProcEnd = std::chrono::steady_clock::now();
    TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex << "] preprocess done in "
                 << std::chrono::duration<double, std::milli>(tPreProcEnd - tPreProcStart).count()
                 << " ms");

    const auto tInferStart = std::chrono::steady_clock::now();
    model.infer();
    const auto tInferEnd = std::chrono::steady_clock::now();
    TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex << "] infer done in "
                 << std::chrono::duration<double, std::milli>(tInferEnd - tInferStart).count()
                 << " ms");

    const auto tPostStart = std::chrono::steady_clock::now();
    model.postprocess();
    const auto tPostEnd = std::chrono::steady_clock::now();
    TRT_LOG_DEBUG("[InferencePool] worker[" << workerIndex << "] postprocess done in "
                 << std::chrono::duration<double, std::milli>(tPostEnd - tPostStart).count()
                 << " ms");

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
    TRT_LOG_DEBUG("[InferencePool] waitIdle: waiting, queue=" << m_tasks.size()
                 << ", active=" << m_activeCount);
    m_cvIdle.wait(lock, [this] { return m_tasks.empty() && m_activeCount == 0; });
    TRT_LOG_DEBUG("[InferencePool] waitIdle: all idle");
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
        TRT_LOG_INFO("[InferencePool] shutdown requested, queue=" << m_tasks.size()
                    << ", active=" << m_activeCount);
    }
    m_cvTask.notify_all();
    m_cvNotFull.notify_all();

    for (std::size_t i = 0; i < m_workers.size(); ++i)
    {
        if (m_workers[i].thread.joinable())
        {
            TRT_LOG_DEBUG("[InferencePool] joining worker[" << i << "]");
            m_workers[i].thread.join();
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