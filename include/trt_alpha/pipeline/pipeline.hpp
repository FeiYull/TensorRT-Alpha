// =============================================================================
//  trt_alpha :: pipeline :: pipeline
// -----------------------------------------------------------------------------
//  Pipeline -- the three-stage pipeline scheduler.
//
//  Threading model:
//    * data-source threads: one per source (N) -- read a frame -> submit -> push
//      to the result queue
//    * inference workers: already inside the pool (M) -- Pipeline does not touch them
//    * render threads: R = PipelineConfig::renderThreads
//      -- take from the result queue -> render -> save / show
//      R == 1 whenever show / exposeProcessed is on; R can be > 1 for
//      "save to disk only" (frames are independent there).
//  N + M + R threads in total.
//
//  Call sequence:
//    Pipeline p(cfg);
//    p.start();               // start N source threads + R render threads
//    p.waitForCompletion();   // block until every source ends and every result is rendered
//
//  Stopping:
//    p.stop();                // request a stop (asynchronous) -- requestStop on every source + close the queues
//    p.waitForCompletion();   // wait for the threads to finish (queued tasks still run to completion)
//
//  Error handling:
//    * a construction / start failure throws
//    * an error inside a source thread (next / submit) -> ERROR log, that source
//      exits, and failed() is set
//    * an error inside the render thread (infer / draw / save / show) -> ERROR
//      log, continue, and failed() is set
//    * errors inside threads have no exception exit (join swallows them), so
//      failed() / firstError() hand "it ran but did not succeed" back to the
//      caller -- usually to decide the process exit code.
//      Convention: **any failure at any step inside a thread implies failed() == true**
//      (only whether something failed matters, not which step).
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

    //! Start: launch N source threads + R render threads. Throws on failure.
    void start();

    //! Request a stop (asynchronous, returns immediately). Thread-safe.
    void stop();

    //! Block until every source ends and every result is rendered.
    void waitForCompletion();

    //! Take one BatchResult from the result queue (blocking). false = end of stream.
    //! Only for use without a render thread (renderer == nullptr).
    bool popResult(core::BatchResult& out);

    //! Take one result from the "post-render result queue" (blocking).
    //! Only for use with a render thread (renderer != nullptr).
    bool popProcessed(core::BatchResult& out);

    //! Whether a render thread was started (renderer != nullptr).
    [[nodiscard]] bool hasRenderer() const noexcept { return m_cfg.renderer != nullptr; }

    [[nodiscard]] bool running() const noexcept { return m_running.load(); }

    //! Whether this run saw any error (any failure while reading / submitting
    //! from a source, inferring, or drawing / saving / showing).
    //! Only complete when read after waitForCompletion().
    [[nodiscard]] bool failed() const noexcept { return m_failed.load(); }

    //! The first error description (empty when there is none). Used by the caller
    //! to summarise it on one line.
    [[nodiscard]] std::string firstError() const;

    //! Total number of frame batches dropped by the result queue (> 0 only when
    //! queueFullPolicy == DropOldest). Convention: **dropping data must be
    //! observable** -- the caller decides from this whether to report
    //! "incomplete results" as a failure.
    [[nodiscard]] std::size_t droppedResults() const noexcept;

    //! Per-source drop count (index = source index). A source that shares a
    //! queue with others reports that queue's count (the queue is the unit of
    //! dropping); the default one-queue-per-source layout makes each entry exact.
    [[nodiscard]] std::vector<std::size_t> droppedResultsPerSource() const noexcept;

private:
    void sourceLoop(std::size_t sourceIndex);
    void renderLoop();

    //! Round-robin pop across every result queue (fair across sources). Blocking:
    //! returns false only when all queues are closed and drained.
    bool popAnyResult(std::future<core::BatchResult>& out);

    //! True once every result queue is closed and empty (the drain condition).
    bool allResultQueuesClosed() const;

    //! Tell the renderer which sources have ended (NO SIGNAL placeholder).
    //! Idempotent: the renderer keeps per-tile state, so calling it every frame
    //! is harmless and needs no extra bookkeeping.
    void notifyEndedSources();

    //! Record an error: set the failed flag + remember the first description
    //! (thread-safe, idempotent).
    void markFailed(const std::string& what);

    void validateConfig();

    PipelineConfig m_cfg;

    // Result queues: future<BatchResult>, one per logical queue (default one per
    // source, see PipelineConfig::sourceToQueue). A vector so a slow source only
    // backs up its own queue instead of blocking every source on a shared one.
    std::vector<std::unique_ptr<core::BoundedQueue<std::future<core::BatchResult>>>> m_resultQueues;

    //! Round-robin cursor into m_resultQueues (renderLoop / popResult). Both are
    //! single consumers that never run concurrently, so no lock is needed.
    std::size_t m_nextQueue = 0;

    // Threads
    std::vector<std::thread> m_sourceThreads;
    std::vector<std::thread> m_renderThreads;

    // Per-source "ended" flags: set by the source thread when it exits (EOF /
    // stop), read by the render thread so it can tell the renderer to show a
    // "NO SIGNAL" placeholder for that source's tile. std::atomic<bool> is
    // neither copyable nor movable, so it cannot live in a std::vector (resize
    // needs a copy); a heap array built with make_unique sidesteps that. Each
    // element is store(false)'d explicitly in start() (the default constructor
    // leaves the value uninitialised, which would be UB to read).
    std::unique_ptr<std::atomic<bool>[]> m_sourceEnded;

    // Lifetime
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_stopRequested{false};

    // Error record: the flag is atomic (lock-free check on the hot path); the
    // first description is protected by m_mutex
    std::atomic<bool> m_failed{false};
    std::string m_firstError;   //!< protected by m_mutex

    // Shared count of "how many sources are still running" (protected by m_mutex).
    // Reaching zero closes the result queue so the render thread can drain and exit.
    mutable std::mutex m_mutex;
    std::size_t m_sourcesRunning = 0;
    std::unique_ptr<core::BoundedQueue<core::BatchResult>> m_processedQueue;
};

}  // namespace trt_alpha::pipeline
