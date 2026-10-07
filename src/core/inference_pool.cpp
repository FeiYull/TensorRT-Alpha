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
        // ---- 阶段 0：先建 1 个共享 Engine（1 engine + N context）----
        TRT_LOG_INFO("[InferencePool] loading shared engine: " << cfg.engine);
        auto sharedEngine = std::make_shared<core::Engine>(cfg.engine);

        // 每个 worker 的 ModelConfig 都带同一个 sharedEngine
        core::ModelConfig workerCfg = cfg;
        workerCfg.sharedEngine = sharedEngine;

        // ---- 阶段 0.5：池级 batch 预检 ----
        // 与模型侧的 core::applyInputShape 调的是同一个 resolveBatch（逻辑零重复，
        // 报错措辞也完全一致），这里先跑一遍只为两件事：
        //   ① 失败点提前到"还没为 N 个 worker 分配显存之前"；
        //   ② 拿到 m_resolvedBatch 供数据源攒批（数据源必须与模型 batch 一致）。
        // 另：模型侧的 applyInputShape 覆盖 13 个模型，yunet 因输入尺寸取自原图不走它
        // （自带一行，见 yunet.cpp）；本预检与 applyInputShape 一起构成完整覆盖。
        // resolveBatch 只判定、不改值 —— 不符即抛，因此 m_resolvedBatch 恒等于请求值。
        for (const auto& t : sharedEngine->ioTensors())
        {
            if (!t.isInput) { continue; }

            // batch 轴由布局决定。配置里给了 input.layout 就用它；
            // 没给则按"秩 >= 4 ⇒ N 在轴 0"这一当前所有模型的规范约定取值。
            // 秩 < 4 且没有布局声明时（CHW / HWC 这类）无法可靠判断，
            // 跳过本预检 —— 权威判定在模型侧的 applyInputShape，那里拿得到布局。
            const int rank = t.shape.nbDims;
            const bool axisKnown = !workerCfg.layout.empty() || rank >= 4;
            if (!axisKnown)
            {
                TRT_LOG_DEBUG("[InferencePool] batch precheck skipped for '"
                              << t.name << "' (rank " << rank
                              << ", no input.layout declared)");
                break;
            }
            const int batchAxis = t.batchAxisIndex(workerCfg.layout);
            m_resolvedBatch = core::resolveBatch(t, workerCfg.batchSize,
                                                 "[InferencePool]",
                                                 workerCfg.maxBatchSize,
                                                 batchAxis).batch;
            break;
        }

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
            worker.model->init(workerCfg);
            TRT_LOG_DEBUG("[InferencePool] worker[" << i << "] init() done");
            m_workers.push_back(std::move(worker));
        }

        // ---- 阶段 1.5：快照 worker[0] 的 I/O 描述 ----
        // 此时模型已 init（形状/类型固定），线程尚未启动；
        // 存副本而非引用，避免外部在 shutdown 后拿到悬垂引用。
        if (!m_workers.empty())
        {
            m_ioDesc = m_workers.front().model->describe();
        }

        // 记下"最终喂给模型的那份配置"（batch 已修正、读取痕迹已产生）。
        // 模型 init 期间对 getXxx 的读取会在 workerCfg.readKeys 上留痕，
        // 所以必须在 init 之后再拷 —— 配置展示（logConfigBox）靠它判定谁真生效。
        m_cfg = workerCfg;

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

    // 容量护栏：显存是按【引擎解析后的 batch】分配的，多喂一批就是对 kernel
    // 越界写（P1-2）。submit 是公开 API，且是外部 Batch 进入系统的唯一入口，
    // 因此这里是最靠前、覆盖最全的收口点 —— 越界一律显式失败，不截断不静默。
    if (m_resolvedBatch > 0 &&
        static_cast<int>(batch.views.size()) > m_resolvedBatch)
    {
        const std::string msg =
            "[InferencePool] batch of " + std::to_string(batch.views.size()) +
            " images exceeds the resolved batch " + std::to_string(m_resolvedBatch) +
            " (buffers are sized by the engine profile; raise [input] batch_size"
            " within the engine range instead)";
        TRT_LOG_ERROR(msg);
        throw std::invalid_argument(msg);
    }

    std::promise<BatchResult> promise;
    auto future = promise.get_future();

    const int batchN = batch.validCount;
    const std::uint64_t taskId = g_taskCounter.fetch_add(1);

    {
        // 所有对 m_tasks 的读（含日志里的 size()）都必须在锁内 ——
        // 锁外读一个正被 worker 线程 pop 的 std::queue 是 data race。
        std::unique_lock<std::mutex> lock(m_mutex);
        TRT_LOG_DEBUG("[InferencePool] submit task #" << taskId
                     << ": validCount=" << batchN << "/" << batch.views.size()
                     << ", queue=" << m_tasks.size() << "/" << m_maxQueueSize);
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
                TRT_LOG_DEBUG("[InferencePool] submit task #" << taskId
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
        }
        m_tasks.emplace(std::move(batch), std::move(promise));
        TRT_LOG_DEBUG("[InferencePool] submit task #" << taskId << " queued, queue="
                     << m_tasks.size());
    }
    m_cvTask.notify_one();

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
    result.frameNames = std::move(batch.frameNames);

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
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_stop)
        {
            // 已有线程在关停：等它 join 完再返回。否则本线程可能在 worker
            // 仍被使用时就析构池对象 → use-after-free。
            m_cvShutdown.wait(lock, [this] { return m_shutdownDone; });
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

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_shutdownDone = true;
    }
    m_cvShutdown.notify_all();

    TRT_LOG_INFO("[InferencePool] shutdown complete");
}

std::size_t InferencePool::pending() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_tasks.size();
}

}  // namespace trt_alpha::core