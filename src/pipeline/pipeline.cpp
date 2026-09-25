// =============================================================================
//  trt_alpha :: pipeline :: pipeline（实现）
// =============================================================================
#include "trt_alpha/pipeline/pipeline.hpp"

#include "trt_alpha/core/bounded_queue.hpp"
#include "trt_alpha/core/inference_pool.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/paths.hpp"
#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/renderer/i_renderer.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace trt_alpha::pipeline {

Pipeline::Pipeline(PipelineConfig cfg)
    : m_cfg(std::move(cfg))
{
    validateConfig();

    m_resultQueue = std::make_unique<core::BoundedQueue<std::future<core::BatchResult>>>(
        m_cfg.resultQueueSize,
        core::BoundedQueue<std::future<core::BatchResult>>::FullPolicy::DropOldest);
}

Pipeline::~Pipeline()
{
    if (m_running.load())
    {
        stop();
        waitForCompletion();
    }
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
    if (m_cfg.renderer == nullptr)
    {
        throw std::runtime_error("Pipeline: renderer is null");
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

    if (m_cfg.resultQueueSize == 0)
    {
        TRT_LOG_WARN("Pipeline: resultQueueSize == 0 -> unbounded (not recommended)");
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

    // ---- 起渲染线程 ----
    TRT_LOG_INFO("Pipeline: launching render thread");          // ← 加
    m_renderThread = std::thread([this] { renderLoop(); });
    TRT_LOG_INFO("Pipeline: render thread launched");           // ← 加

    // ---- 起源线程 ----
    m_sourceThreads.reserve(m_cfg.sources.size());
    for (std::size_t i = 0; i < m_cfg.sources.size(); ++i)
    {
        TRT_LOG_INFO("Pipeline: launching source thread " << i); // ← 加
        m_sourceThreads.emplace_back([this, i] { sourceLoop(i); });
    }
    TRT_LOG_INFO("Pipeline: all source threads launched");      // ← 加

    TRT_LOG_INFO("Pipeline: started (N sources + 1 render = "
                 << (m_cfg.sources.size() + 1) << " threads)");
}

void Pipeline::sourceLoop(std::size_t sourceIndex)
{
    TRT_LOG_INFO("Pipeline: source[" << sourceIndex << "] thread entered");   // ← 加

    TRT_LOG_INFO("Pipeline: source[" << sourceIndex
                 << "] accessing sources[" << sourceIndex << "]");             // ← 加
    auto& source = *m_cfg.sources[sourceIndex];

    TRT_LOG_INFO("Pipeline: source[" << sourceIndex
                 << "] accessing pools[" << m_cfg.sourceToPool[sourceIndex] << "]");  // ← 加
    core::InferencePool* pool = m_cfg.pools[m_cfg.sourceToPool[sourceIndex]];

    TRT_LOG_INFO("Pipeline: source[" << sourceIndex << "] '" << source.typeName()
                 << "' using pool[" << m_cfg.sourceToPool[sourceIndex] << "]");

    int iterCount = 0;                                                         // ← 加

    while (!m_stopRequested.load())
    {
        ++iterCount;                                                           // ← 加
        TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                     << "] iteration " << iterCount << ": calling next()");    // ← 加

        core::Batch batch;
        try
        {
            if (!source.next(batch))
            {
                TRT_LOG_INFO("Pipeline: source[" << sourceIndex
                             << "] next() returned false, breaking");          // ← 加
                break;
            }
        }
        catch (const std::exception& e)
        {
            TRT_LOG_ERROR("Pipeline: source[" << sourceIndex << "] next() failed: "
                          << e.what());
            break;
        }

        TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                     << "] got batch, views=" << batch.views.size()
                     << " validCount=" << batch.validCount
                     << " buffer=" << (batch.buffer ? "ok" : "null"));         // ← 加

        // 提交到池
        std::future<core::BatchResult> future;
        try
        {
            TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                         << "] submitting to pool");                           // ← 加
            future = pool->submit(std::move(batch));
            TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                         << "] submit returned");                              // ← 加
        }
        catch (const std::exception& e)
        {
            TRT_LOG_ERROR("Pipeline: source[" << sourceIndex
                          << "] submit failed: " << e.what());
            break;
        }

        // 推到结果队列
        TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                     << "] pushing future to result queue");                   // ← 加
        bool dropped = false;
        if (!m_resultQueue->push(std::move(future), &dropped))
        {
            TRT_LOG_WARN("Pipeline: source[" << sourceIndex
                         << "] result queue closed");                          // ← 加
            break;
        }
        if (dropped)
        {
            TRT_LOG_WARN("Pipeline: result queue full (" << m_cfg.resultQueueSize
                          << "), dropped oldest future");
        }
        TRT_LOG_DEBUG("Pipeline: source[" << sourceIndex
                     << "] pushed future, loop again");                        // ← 加
    }

    TRT_LOG_INFO("Pipeline: source[" << sourceIndex << "] exiting");

    // 通知"有一个源结束"
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_sourcesRunning > 0)
        {
            --m_sourcesRunning;
        }
        if (m_sourcesRunning == 0)
        {
            m_resultQueue->close();
        }
    }
    m_cvDone.notify_all();
}

void Pipeline::renderLoop()
{
    TRT_LOG_INFO("Pipeline: render loop started, entering while");   // ← 改（加 "entering while"）

    int loopCount = 0;                                               // ← 加
    while (true)
    {
        ++loopCount;                                                 // ← 加
        TRT_LOG_DEBUG("Pipeline: render loop iteration " << loopCount
                     << ": waiting for future");                     // ← 加

        std::future<core::BatchResult> future;
        if (!m_resultQueue->pop(future))
        {
            TRT_LOG_INFO("Pipeline: render loop pop() returned false, exiting");  // ← 加
            break;
        }

        TRT_LOG_DEBUG("Pipeline: render loop got future, valid=" << future.valid());  // ← 加

        if (!future.valid())
        {
            TRT_LOG_WARN("Pipeline: future invalid, skipping");       // ← 加
            continue;
        }

        // 等推理完成
        TRT_LOG_DEBUG("Pipeline: render loop waiting future.get()");  // ← 加
        core::BatchResult result;
        try
        {
            result = future.get();
        }
        catch (const std::exception& e)
        {
            TRT_LOG_ERROR("Pipeline: inference failed: " << e.what());
            continue;
        }

        // ---- 诊断日志 ----
        TRT_LOG_INFO("Pipeline: got result  views=" << result.views.size()
                     << " validCount=" << result.validCount
                     << " detections=" << result.detections.size()
                     << " segmentations=" << result.segmentations.size()
                     << " classifications=" << result.classifications.size()
                     << " buffer=" << (result.buffer ? "ok" : "null"));  // ← 加

        const std::size_t n =
            std::min<std::size_t>(result.views.size(),
                static_cast<std::size_t>(std::max(0, result.validCount)));
        for (std::size_t i = 0; i < n; ++i)
        {
            TRT_LOG_INFO("Pipeline: view[" << i << "] data="
                         << static_cast<const void*>(result.views[i].data)
                         << " w=" << result.views[i].width
                         << " h=" << result.views[i].height
                         << " stride=" << result.views[i].stride);        // ← 加
        }
        // ---- 诊断日志结束 ----

        // 画
        TRT_LOG_DEBUG("Pipeline: calling drawResult");                    // ← 加
        try
        {
            m_cfg.renderer->drawResult(result, m_cfg.classNames);
            TRT_LOG_DEBUG("Pipeline: drawResult done");                   // ← 加
        }
        catch (const std::exception& e)
        {
            TRT_LOG_ERROR("Pipeline: drawResult failed: " << e.what());
            continue;
        }

        if (m_cfg.saveEnabled)
        {
            TRT_LOG_DEBUG("Pipeline: calling save");                      // ← 加
            try
            {
                m_cfg.renderer->save(result, m_cfg.saveDir);
            }
            catch (const std::exception& e)
            {
                TRT_LOG_ERROR("Pipeline: save failed: " << e.what());
            }
        }
        // ... show 部分不变
    }

    TRT_LOG_INFO("Pipeline: render loop exiting");
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
    m_resultQueue->close();

    // 等渲染线程结束
    if (m_renderThread.joinable())
    {
        m_renderThread.join();
    }

    m_running.store(false);
    TRT_LOG_INFO("Pipeline: all threads joined");
}

}  // namespace trt_alpha::pipeline