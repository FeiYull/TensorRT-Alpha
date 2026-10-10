// =============================================================================
//  trt_alpha :: pipeline :: pipeline
// -----------------------------------------------------------------------------
//  Pipeline -- the three-stage pipeline scheduler.
//
//  Threading model:
//    * data-source threads: one per source (N) -- read a frame -> submit -> push
//      to the result queue
//    * inference workers: already inside the pool (M) -- Pipeline does not touch them
//    * render thread: 1 -- take from the result queue -> render -> save / show
//  N + M + 1 threads in total.
//
//  Call sequence:
//    Pipeline p(cfg);
//    p.start();               // start N source threads + 1 render thread
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

    //! Start: launch N source threads + 1 render thread. Throws on failure.
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

private:
    void sourceLoop(std::size_t sourceIndex);
    void renderLoop();

    //! Record an error: set the failed flag + remember the first description
    //! (thread-safe, idempotent).
    void markFailed(const std::string& what);

    void validateConfig();

    PipelineConfig m_cfg;

    // Result queue: future<BatchResult>
    std::unique_ptr<core::BoundedQueue<std::future<core::BatchResult>>> m_resultQueue;

    // Threads
    std::vector<std::thread> m_sourceThreads;
    std::thread m_renderThread;

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
