// =============================================================================
//  trt_alpha :: pipeline :: pipeline（实现）
// =============================================================================
#include "trt_alpha/pipeline/pipeline.hpp"

#include "trt_alpha/core/bounded_queue.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/inference_pool.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/paths.hpp"
#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/renderer/i_renderer.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace trt_alpha::pipeline {

namespace {

//! Auto render-thread count for a "save to disk only" pipeline.
//! Encoding an annotated frame (imwrite -> JPEG) is per-frame independent and
//! dominates --save wall time, so it scales nearly linearly with the thread
//! count until the page cache / writeback becomes the limit.
//! Measured on this box (32 threads, 1080p, 3262 frames, wall clock): no --save
//! 15.0 s, --save at rt=4 21.2 s, rt=6 16.0 s, rt=8 16.2 s, rt=12 16.4 s,
//! rt=16 18.8 s. The old cap of 4 was therefore the worst possible choice --
//! 8 is where the curve flattens, and the frame rate there matches the
//! no-save one. Past 8 the threads only fight over writeback.
//! The inference worker needs one core (it is mostly CUDA launches), so even
//! hw/4 render threads leave plenty behind.
int autoRenderThreads()
{
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0)
    {
        hw = 4;   // unknown -> assume a small box
    }
    return std::min(8, std::max(1, static_cast<int>(hw / 4)));
}

}  // namespace

Pipeline::Pipeline(PipelineConfig cfg)
    : m_cfg(std::move(cfg))
{
    validateConfig();

    // 结果队列 + 处理后队列都用配置里的策略（默认 Block：批处理不丢帧）。
    // 注意 Block 的安全性由 stop() / waitForCompletion() 保证 —— 它们会在 join
    // 之前 close() 队列，让阻塞在 push 上的线程有出口（否则会挂死）。
    // 结果队列按 sourceToQueue 拆成一条或多条（默认每 source 一条）。
    std::size_t queueCount = 1;
    for (std::size_t k : m_cfg.sourceToQueue)
    {
        queueCount = std::max(queueCount, k + 1);
    }
    m_resultQueues.reserve(queueCount);
    for (std::size_t i = 0; i < queueCount; ++i)
    {
        m_resultQueues.push_back(
            std::make_unique<core::BoundedQueue<std::future<core::BatchResult>>>(
                m_cfg.resultQueueSize, m_cfg.queueFullPolicy));
    }

    m_processedQueue = std::make_unique<core::BoundedQueue<core::BatchResult>>(
        m_cfg.resultQueueSize, m_cfg.queueFullPolicy);
}

Pipeline::~Pipeline()
{
    if (m_running.load())
    {
        stop();
        waitForCompletion();
    }
}

void Pipeline::markFailed(const std::string& what)
{
    m_failed.store(true);
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_firstError.empty())
    {
        m_firstError = what;
    }
}

std::string Pipeline::firstError() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_firstError;
}

void Pipeline::validateConfig()
{
    if (m_cfg.sources.empty())
    {
        throw std::runtime_error("Pipeline: no sources");
    }
    if (m_cfg.pools.empty())
    {
        throw std::runtime_error("Pipeline: no pools");
    }

    // sourceToPool 校验
    if (m_cfg.sourceToPool.empty())
    {
        // 空 = 所有源用 pools[0]
        m_cfg.sourceToPool.assign(m_cfg.sources.size(), 0);
    }
    else if (m_cfg.sourceToPool.size() != m_cfg.sources.size())
    {
        throw std::runtime_error(
            "Pipeline: sourceToPool size (" +
            std::to_string(m_cfg.sourceToPool.size()) +
            ") != sources size (" + std::to_string(m_cfg.sources.size()) + ")");
    }

    for (std::size_t i = 0; i < m_cfg.sourceToPool.size(); ++i)
    {
        if (m_cfg.sourceToPool[i] >= m_cfg.pools.size())
        {
            throw std::runtime_error(
                "Pipeline: sourceToPool[" + std::to_string(i) + "] = " +
                std::to_string(m_cfg.sourceToPool[i]) + " >= pools size " +
                std::to_string(m_cfg.pools.size()));
        }
    }

    // sourceToQueue 校验（对标 sourceToPool）：空 = 每 source 一条独立队列。
    // 队列本身是泛型模板，数量在构造函数里据 max(sourceToQueue)+1 创建。
    if (m_cfg.sourceToQueue.empty())
    {
        m_cfg.sourceToQueue.resize(m_cfg.sources.size());
        for (std::size_t i = 0; i < m_cfg.sources.size(); ++i)
        {
            m_cfg.sourceToQueue[i] = i;
        }
    }
    else if (m_cfg.sourceToQueue.size() != m_cfg.sources.size())
    {
        throw std::runtime_error(
            "Pipeline: sourceToQueue size (" +
            std::to_string(m_cfg.sourceToQueue.size()) +
            ") != sources size (" + std::to_string(m_cfg.sources.size()) + ")");
    }

    if (m_cfg.resultQueueSize == 0)
    {
        TRT_LOG_WARN("Pipeline: resultQueueSize == 0 -> unbounded (not recommended)");
    }

    // Block 策略下没有消费者会直接挂死（源线程写到满就永远等下去）。
    // 渲染线程存在时它自己就是消费者；否则责任在调用方（必须持续 popResult）。
    if (m_cfg.queueFullPolicy == core::QueueFullPolicy::Block &&
        m_cfg.renderer == nullptr)
    {
        TRT_LOG_WARN("Pipeline: queueFullPolicy=Block without a renderer; the caller "
                     "must keep draining popResult() or the source thread will block");
    }

    // 渲染线程数：0 = auto。落盘是"顺序无关"的纯 CPU 编码，每帧独立，可以并行；
    // --show / exposeProcessed 要求严格按序（cv::imshow 也非线程安全），必须单线程。
    // 未指定（0）且没有渲染器时保持 0 —— 没有渲染线程可起。
    if (m_cfg.renderer == nullptr)
    {
        if (m_cfg.renderThreads != 0)
        {
            TRT_LOG_WARN("Pipeline: no renderer, renderThreads=" << m_cfg.renderThreads
                         << " ignored -> 0");
            m_cfg.renderThreads = 0;
        }
    }
    else if (m_cfg.showEnabled || m_cfg.exposeProcessed)
    {
        if (m_cfg.renderThreads != 1)
        {
            TRT_LOG_WARN("Pipeline: showEnabled / exposeProcessed needs the frames in "
                         "order, forcing render_threads=" << m_cfg.renderThreads
                         << " -> 1");
            m_cfg.renderThreads = 1;
        }
    }
    else if (m_cfg.renderThreads < 0)
    {
        TRT_LOG_WARN("Pipeline: renderThreads=" << m_cfg.renderThreads
                     << " is invalid, falling back to auto");
        m_cfg.renderThreads = autoRenderThreads();
    }
    else if (m_cfg.renderThreads == 0)
    {
        m_cfg.renderThreads = m_cfg.saveEnabled ? autoRenderThreads() : 1;
    }

    if (m_cfg.renderer != nullptr && m_cfg.renderThreads > 1)
    {
        TRT_LOG_INFO("Pipeline: render_threads=" << m_cfg.renderThreads
                     << " (frame order on disk is not guaranteed)");
    }
}

void Pipeline::start()
{
    if (m_running.load())
    {
        throw std::logic_error("Pipeline: already running");
    }

    TRT_LOG_INFO("Pipeline: starting with " << m_cfg.sources.size()
                 << " source(s) and " << m_cfg.pools.size() << " pool(s)");

    m_stopRequested.store(false);
    m_running.store(true);
    m_sourcesRunning = m_cfg.sources.size();

    // Per-source ended flags (atomic<bool> has no value-initialised default, so
    // store(false) explicitly; reading an uninitialised atomic is UB). atomic<bool>
    // is non-copyable/non-movable, hence a raw unique_ptr array instead of vector.
    const std::size_t sourceCount = m_cfg.sources.size();
    m_sourceEnded = std::make_unique<std::atomic<bool>[]>(sourceCount);
    for (std::size_t i = 0; i < sourceCount; ++i)
    {
        m_sourceEnded[i].store(false);
    }

    // ---- 起渲染线程（仅 renderer != nullptr 时）----
    if (m_cfg.renderer != nullptr)
    {
        const std::size_t n =
            (m_cfg.renderThreads > 0) ? static_cast<std::size_t>(m_cfg.renderThreads)
                                      : std::size_t{1};
        TRT_LOG_INFO("Pipeline: launching " << n << " render thread(s)");
        m_renderThreads.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            m_renderThreads.emplace_back([this] { renderLoop(); });
        }
        TRT_LOG_INFO("Pipeline: render thread(s) launched");
    }
    else
    {
        TRT_LOG_INFO("Pipeline: no renderer, skip render thread");
    }

    // ---- 起源线程 ----
    m_sourceThreads.reserve(m_cfg.sources.size());
    for (std::size_t i = 0; i < m_cfg.sources.size(); ++i)
    {
        TRT_LOG_INFO("Pipeline: launching source thread " << i);
        m_sourceThreads.emplace_back([this, i] { sourceLoop(i); });
    }
    TRT_LOG_INFO("Pipeline: all source threads launched");

    const std::size_t renderThreads = m_renderThreads.size();
    TRT_LOG_INFO("Pipeline: started (" << m_cfg.sources.size() << " source + "
                 << renderThreads << " render = "
                 << (m_cfg.sources.size() + renderThreads) << " threads)");

    // 窗口名是拼出来的（"trt_alpha:<net>"），配置框里显示的是 ini 原值，
    // 所以这里单独打一行，出问题时能直接看到最终标题。
    if (m_cfg.showEnabled)
    {
        TRT_LOG_INFO("Pipeline: show window title = '" << m_cfg.showWindow << "'");
    }
}

void Pipeline::sourceLoop(std::size_t sourceIndex)
{
    TRT_LOG_INFO("Pipeline: source[" << sourceIndex << "] thread entered");

    auto& source = *m_cfg.sources[sourceIndex];
    core::InferencePool* pool = m_cfg.pools[m_cfg.sourceToPool[sourceIndex]];

    TRT_LOG_INFO("Pipeline: source[" << sourceIndex << "] '" << source.typeName()
                 << "' using pool[" << m_cfg.sourceToPool[sourceIndex] << "]");

    int iterCount = 0;

    while (!m_stopRequested.load())
    {
        ++iterCount;
        TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                     << "] iteration " << iterCount << ": calling next()");

        core::Batch batch;
        try
        {
            if (!source.next(batch))
            {
                TRT_LOG_INFO("Pipeline: source[" << sourceIndex
                             << "] next() returned false, breaking");
                break;
            }
        }
        catch (const std::exception& e)
        {
            TRT_LOG_ERROR("Pipeline: source[" << sourceIndex << "] next() failed: "
                          << e.what());
            markFailed("source[" + std::to_string(sourceIndex) + "] next: " + e.what());
            break;
        }

        TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                     << "] got batch, views=" << batch.views.size()
                     << " validCount=" << batch.validCount
                     << " buffer=" << (batch.buffer ? "ok" : "null"));

        // 提交到池
        std::future<core::BatchResult> future;
        try
        {
            TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                         << "] submitting to pool");
            future = pool->submit(std::move(batch));
            TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                         << "] submit returned");
        }
        catch (const std::exception& e)
        {
            TRT_LOG_ERROR("Pipeline: source[" << sourceIndex
                          << "] submit failed: " << e.what());
            markFailed("source[" + std::to_string(sourceIndex) + "] submit: " + e.what());
            break;
        }

        // 推到该源的队列（sourceToQueue[sourceIndex]）。每 source 一条队列时，
        // 慢源只堵自己，不阻塞其它源。
        TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                     << "] pushing future to result queue "
                     << m_cfg.sourceToQueue[sourceIndex]);
        bool dropped = false;
        auto& queue = *m_resultQueues[m_cfg.sourceToQueue[sourceIndex]];
        if (!queue.push(std::move(future), &dropped))
        {
            TRT_LOG_WARN("Pipeline: source[" << sourceIndex
                         << "] result queue closed");
            break;
        }
        if (dropped)
        {
            // 丢帧必须留痕，且必须是默认可见的 WARN 级（DEBUG 在 Release 被编译掉）。
            // 逐帧刷屏没意义：首次报一条，之后每 100 条报一次，末尾再汇总一次。
            const std::size_t total = queue.droppedTotal();
            if (total == 1 || total % 100 == 0)
            {
                TRT_LOG_WARN("Pipeline: source[" << sourceIndex
                             << "] result queue full (size=" << m_cfg.resultQueueSize
                             << "), dropped the oldest result(s); total dropped=" << total
                             << " - set queue_full_policy=block to never drop");
            }
        }
        TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                     << "] pushed future, loop again");
    }

    TRT_LOG_INFO("Pipeline: source[" << sourceIndex << "] exiting");

    // 标记该源结束：渲染线程据此把对应 tile 切成 "NO SIGNAL"。
    if (sourceIndex < m_cfg.sources.size())
    {
        m_sourceEnded[sourceIndex].store(true);
    }

    // 通知"有一个源结束"
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_sourcesRunning > 0)
        {
            --m_sourcesRunning;
        }
        if (m_sourcesRunning == 0)
        {
            // 全部源结束 → 关闭所有结果队列，让渲染线程 / popResult 排空后退出。
            for (auto& q : m_resultQueues)
            {
                q->close();
            }
        }
    }
}

void Pipeline::renderLoop()
{
    TRT_LOG_INFO("Pipeline: render loop started, entering while");

    int loopCount = 0;
    while (true)
    {
        ++loopCount;
        TRT_LOG_DEBUG("Pipeline: render loop iteration " << loopCount
                     << ": waiting for future");

        std::future<core::BatchResult> future;
        if (!popAnyResult(future))
        {
            TRT_LOG_INFO("Pipeline: render loop pop() returned false, exiting");
            break;
        }

        // 先通知"已结束的源"（NO SIGNAL），再画本帧 —— 先播完的路不会冻结在
        // 最后一帧，而是立刻切成 NO SIGNAL。
        notifyEndedSources();

        TRT_LOG_DEBUG("Pipeline: render loop got future, valid=" << future.valid());

        if (!future.valid())
        {
            TRT_LOG_WARN("Pipeline: future invalid, skipping");
            continue;
        }

        // 等推理完成
        TRT_LOG_DEBUG("Pipeline: render loop waiting future.get()");
        core::BatchResult result;
        try
        {
            result = future.get();
        }
        catch (const std::exception& e)
        {
            TRT_LOG_ERROR("Pipeline: inference failed: " << e.what());
            markFailed(std::string("inference: ") + e.what());
            continue;
        }

        TRT_LOG_DEBUG("Pipeline: got result  views=" << result.views.size()
                     << " validCount=" << result.validCount
                     << " detections=" << result.detections.size()
                     << " segmentations=" << result.segmentations.size()
                     << " classifications=" << result.classifications.size()
                     << " buffer=" << (result.buffer ? "ok" : "null"));

        const std::size_t n =
            std::min<std::size_t>(result.views.size(),
                static_cast<std::size_t>(std::max(0, result.validCount)));
        for (std::size_t i = 0; i < n; ++i)
        {
            TRT_LOG_DEBUG("Pipeline: view[" << i << "] data="
                         << static_cast<const void*>(result.views[i].data)
                         << " w=" << result.views[i].width
                         << " h=" << result.views[i].height
                         << " stride=" << result.views[i].stride);
        }

        // 渲染是 in-place 的（直接往 result.buffer 的像素上画框）。若调用方要读
        // 结果（exposeProcessed），就不能让渲染污染原图 —— 先把像素深拷一份，让
        // 渲染画在副本上，画完再把原 buffer / views 换回来。任务结果
        // （boxes / masks / keypoints / depth / ...）全程不动，零拷贝。
        const bool keepRaw = m_cfg.exposeProcessed && m_processedQueue != nullptr;
        std::shared_ptr<core::Buffer>    savedBuffer;
        std::vector<const std::uint8_t*> savedViewData;
        if (keepRaw && result.buffer != nullptr)
        {
            savedBuffer = result.buffer;
            savedViewData.reserve(result.views.size());
            for (const auto& v : result.views) { savedViewData.push_back(v.data); }

            const std::uint8_t* oldBase = result.buffer->data();
            std::shared_ptr<core::Buffer> renderBuffer = core::Buffer::fromHostData(
                oldBase,
                result.buffer->width(), result.buffer->height(),
                result.buffer->channels(), result.buffer->stride(),
                result.buffer->dtype());
            for (auto& v : result.views)
            {
                v.data = renderBuffer->data() + (v.data - oldBase);
            }
            result.buffer = std::move(renderBuffer);

            // Hand back BOTH: keep the drawn-on copy around. drawResult / save /
            // show below all work on this copy, so once they are done renderedViews
            // holds the annotated pixels while views gets restored to the raw
            // source. Without this the rendered frame would be thrown away and only
            // the saved file would be able to show it.
            result.renderedBuffer = result.buffer;
            result.renderedViews  = result.views;
        }

        // 画
        TRT_LOG_DEBUG("Pipeline: calling drawResult");
        try
        {
            m_cfg.renderer->drawResult(result, m_cfg.classNames, m_cfg.renderOptions);
            TRT_LOG_DEBUG("Pipeline: drawResult done");
        }
        catch (const std::exception& e)
        {
            TRT_LOG_ERROR("Pipeline: drawResult failed: " << e.what());
            markFailed(std::string("draw: ") + e.what());
            continue;
        }

        if (m_cfg.saveEnabled)
        {
            TRT_LOG_DEBUG("Pipeline: calling save");
            try
            {
                m_cfg.renderer->save(result, m_cfg.saveDir);
            }
            catch (const std::exception& e)
            {
                TRT_LOG_ERROR("Pipeline: save failed: " << e.what());
                markFailed(std::string("save: ") + e.what());
            }
        }
        if (m_cfg.showEnabled)
        {
            TRT_LOG_DEBUG("Pipeline: calling show");
            try
            {
                m_cfg.renderer->show(result, m_cfg.showWindow);
            }
            catch (const std::exception& e)
            {
                TRT_LOG_ERROR("Pipeline: show failed: " << e.what());
                markFailed(std::string("show: ") + e.what());
            }
        }

        // 窗口被用户关闭（右上角 X / ESC / q）→ 安全停止整个管线并退出渲染
        // 循环。之前这里没有出口：present() 只是跳过，渲染线程继续 pop、源
        // 线程继续解码，程序一直跑到视频结束才退，表现为"点关闭卡死"。
        if (m_cfg.showEnabled && m_cfg.renderer != nullptr &&
            m_cfg.renderer->shouldClose())
        {
            TRT_LOG_INFO("Pipeline: display window closed, stopping the run");
            stop();
            break;
        }

        // 还原原图：渲染只借用了像素克隆，swap 回来把【原图】留给调用方。
        if (keepRaw && savedBuffer != nullptr)
        {
            result.buffer = std::move(savedBuffer);
            for (std::size_t vi = 0; vi < savedViewData.size(); ++vi)
            {
                result.views[vi].data = savedViewData[vi];
            }
        }

        // 把结果推给"处理后队列"（供用户 popProcessed 取）。内容是【原图】——
        // 渲染只影响落盘 / 显示。仅当调用方声明要消费时才推：没人消费还推，
        // Block 策略下渲染线程会写满队列后永久阻塞（Drop 策略下则纯属白丢）。
        if (keepRaw)
        {
            // 处理后队列也可能满（DropOldest）——丢帧必须留痕，接住 droppedOut
            // 并计入 droppedResults()，绝不静默。首次报一条，之后每 100 条一次。
            bool procDropped = false;
            m_processedQueue->push(std::move(result), &procDropped);
            if (procDropped)
            {
                const std::size_t total = m_processedQueue->droppedTotal();
                if (total == 1 || total % 100 == 0)
                {
                    TRT_LOG_WARN("Pipeline: processed queue full (size="
                                 << m_cfg.resultQueueSize
                                 << "), dropped the oldest processed result(s); total dropped="
                                 << total << " - drain popProcessed() faster");
                }
            }
        }
    }

    // 所有源结束、结果队列已关闭。窗口（若开着）直接关闭并释放 —— 用户要求
    // "跑完所有视频自动关闭"，不再常驻等手动关闭。NO SIGNAL 的中间态显示由上面
    // 每帧 pop 后的 notifyEndedSources() 保证：先播完的路停在 NO SIGNAL，直到
    // 全部播完、窗口随之关闭。渲染线程退出后，窗口随 renderer 析构（glfwDestroyWindow）
    // 一起释放。
    if (m_cfg.showEnabled && m_cfg.renderer != nullptr)
    {
        notifyEndedSources();
        TRT_LOG_INFO("Pipeline: all sources ended; closing the window");
    }

    if (m_processedQueue != nullptr)
    {
        m_processedQueue->close();
    }

    TRT_LOG_INFO("Pipeline: render loop exiting");
}

void Pipeline::notifyEndedSources()
{
    if (m_cfg.renderer == nullptr)
    {
        return;
    }
    for (std::size_t i = 0; i < m_cfg.sources.size(); ++i)
    {
        if (m_sourceEnded[i].load())
        {
            m_cfg.renderer->onSourceEnded(static_cast<int>(i));
        }
    }
}

void Pipeline::stop()
{
    if (!m_running.load())
    {
        return;
    }
    if (m_stopRequested.exchange(true))
    {
        return;   // 已经请求过
    }

    TRT_LOG_INFO("Pipeline: stop requested");

    // 通知所有源停止
    for (auto& s : m_cfg.sources)
    {
        s->requestStop();
    }

    // 关掉结果队列：Block 策略下源线程可能正阻塞在 push 上，
    // 不 close 它就没机会回到 while 顶部看到 m_stopRequested → join 会挂死。
    // close 之后队列里已有的元素仍可被渲染线程排空，不会丢。
    for (auto& q : m_resultQueues)
    {
        q->close();
    }
}

void Pipeline::waitForCompletion()
{
    // 等所有源线程结束
    for (auto& t : m_sourceThreads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    m_sourceThreads.clear();

    // 源都结束了 → 确保队列关闭
    for (auto& q : m_resultQueues)
    {
        q->close();
    }

    // 渲染线程可能在 Block 策略下阻塞于 m_processedQueue 的 push（消费者跟不上）。
    // 必须在 join 之前 close，否则会永久挂死。
    if (m_processedQueue != nullptr)
    {
        m_processedQueue->close();
    }

    // 等渲染线程结束（"save 独占"配置下可能不止一个）
    for (auto& t : m_renderThreads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    m_renderThreads.clear();

    m_running.store(false);

    const std::vector<std::size_t> perSource = droppedResultsPerSource();
    std::size_t dropped = 0;
    for (std::size_t c : perSource)
    {
        dropped += c;
    }
    if (dropped > 0)
    {
        TRT_LOG_WARN("Pipeline: finished with " << dropped
                     << " dropped result(s) - queue_full_policy="
                     << (m_cfg.queueFullPolicy == core::QueueFullPolicy::Block
                             ? "block"
                             : (m_cfg.queueFullPolicy == core::QueueFullPolicy::DropNewest
                                    ? "drop_newest" : "drop_oldest"))
                     << ", result_queue_size=" << m_cfg.resultQueueSize);
        // 按源明细：默认每 source 一条队列时精确到路；共享队列时同队列多源会重复。
        for (std::size_t i = 0; i < perSource.size(); ++i)
        {
            if (perSource[i] > 0)
            {
                TRT_LOG_WARN("Pipeline:   source[" << i << "] dropped "
                             << perSource[i] << " result(s)");
            }
        }
    }
    TRT_LOG_INFO("Pipeline: all threads joined");
}

std::size_t Pipeline::droppedResults() const noexcept
{
    std::size_t total = 0;
    for (const auto& q : m_resultQueues)
    {
        total += q->droppedTotal();
    }
    // 处理后队列也可能丢帧（exposeProcessed + DropOldest），一并计入，绝不静默。
    if (m_processedQueue != nullptr)
    {
        total += m_processedQueue->droppedTotal();
    }
    return total;
}

std::vector<std::size_t> Pipeline::droppedResultsPerSource() const noexcept
{
    std::vector<std::size_t> out(m_cfg.sources.size(), 0);
    for (std::size_t i = 0; i < m_cfg.sources.size(); ++i)
    {
        const std::size_t q = m_cfg.sourceToQueue[i];
        if (q < m_resultQueues.size())
        {
            out[i] = m_resultQueues[q]->droppedTotal();
        }
    }
    return out;
}

bool Pipeline::popAnyResult(std::future<core::BatchResult>& out)
{
    while (true)
    {
        const std::size_t n = m_resultQueues.size();
        for (std::size_t round = 0; round < n; ++round)
        {
            const std::size_t idx = (m_nextQueue + round) % n;
            if (m_resultQueues[idx]->tryPop(out))
            {
                m_nextQueue = (idx + 1) % n;
                return true;
            }
        }
        if (allResultQueuesClosed())
        {
            return false;
        }
        // 全空但未关闭：让出 CPU，避免忙等。500µs 对 30fps 渲染可忽略。
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

bool Pipeline::allResultQueuesClosed() const
{
    for (const auto& q : m_resultQueues)
    {
        if (!q->closed() || q->size() > 0)
        {
            return false;
        }
    }
    return true;
}

bool Pipeline::popResult(core::BatchResult& out)
{
    std::future<core::BatchResult> fut;
    if (!popAnyResult(fut))
    {
        return false;
    }
    if (!fut.valid())
    {
        return false;
    }

    try
    {
        out = fut.get();
    }
    catch (const std::exception& e)
    {
        TRT_LOG_ERROR("Pipeline::popResult: inference failed: " << e.what());
        markFailed(std::string("inference: ") + e.what());
        return false;
    }
    return true;
}

bool Pipeline::popProcessed(core::BatchResult& out)
{
    if (m_processedQueue == nullptr)
    {
        return false;
    }
    return m_processedQueue->pop(out);
}

}  // namespace trt_alpha::pipeline
