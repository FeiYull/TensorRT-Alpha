// =============================================================================
//  trt_alpha :: core :: inference_pool
// -----------------------------------------------------------------------------
//  The inference pool (one pool per model): N workers, each with its own context
//  and stream. It follows TensorRT's official concurrency model:
//    * each IExecutionContext may be used by only one thread
//    * different threads use different contexts and streams to infer concurrently
//    * ICudaEngine is thread-safe and can be shared by multiple contexts
//
//  Call sequence:
//    pool.submit(batch) -> future<BatchResult>
//    future.get() blocks until that batch completes
//
//  Error handling:
//    * a construction failure throws std::runtime_error
//    * submitting to a closed pool throws ThreadPoolStopped (synchronously)
//    * when the queue is full, submit blocks (Block policy) until there is room
//    * an error inside a worker is thrown by future.get()
//
//  Model creation:
//    the caller passes a Factory (typically ModelRegistry::create(name)).
//    * tests / the CLI do not need to include concrete model headers
//    * classification / segmentation / future task types are all treated the same
//
//  Resource management:
//    * each worker owns one IModel instance exclusively
//    * an IModel's CUDA stream belongs to the model; the pool does not touch it
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

//! Thrown synchronously when a task is submitted to an already closed pool.
class ThreadPoolStopped : public std::runtime_error
{
public:
    ThreadPoolStopped() : std::runtime_error("inference pool has been stopped") {}
};

//! The inference pool.
class InferencePool
{
public:
    //! Model factory: called once per worker, returning an independent instance
    //! (each initializes itself).
    //! Typical implementation: [] { return ModelRegistry::instance().create("yolov8"); }
    using Factory = std::function<std::unique_ptr<IModel>()>;

    //! Constructor: calls factory to create `workers` independent IModel instances.
    //! workers == 0 uses min(hardware_concurrency, 8).
    //! maxQueueSize == 0 makes the queue unbounded (not recommended).
    //! An empty factory / a nullptr return / a failed init -> throws std::runtime_error.
    InferencePool(const ModelConfig& cfg,
                  Factory factory,
                  std::size_t workers = 0,
                  std::size_t maxQueueSize = 16);

    ~InferencePool();

    InferencePool(const InferencePool&) = delete;
    InferencePool& operator=(const InferencePool&) = delete;
    InferencePool(InferencePool&&) = delete;
    InferencePool& operator=(InferencePool&&) = delete;

    //! Submit one batch (asynchronous).
    //! - returns a future immediately
    //! - blocks when the queue is full (Block policy) until there is room
    //! - throws ThreadPoolStopped synchronously when the pool is closed
    [[nodiscard]] std::future<BatchResult> submit(Batch batch);

    //! Wait for every submitted task to finish (without closing the pool).
    void waitIdle();

    //! Shutdown: reject new tasks, let the queue drain, then join every worker.
    //! Idempotent.
    void shutdown();

    [[nodiscard]] std::size_t size() const noexcept { return m_workers.size(); }
    [[nodiscard]] std::size_t pending() const;

    //! The input batch after correction to the engine's real capability (static =
    //! a fixed value; dynamic must lie within [min, max], and an out-of-range
    //! value already throws during construction). Data sources must use it to
    //! accumulate batches, so they agree with the model / engine.
    [[nodiscard]] int resolvedBatch() const noexcept { return m_resolvedBatch; }

    //! The engine I/O tensor descriptions of the first worker (initialized during
    //! construction, read-only afterwards). Used by the config display (the
    //! [resolved] section of logConfigBox). What is returned is an internal
    //! [copy]; no caller holds a reference into a worker's model.
    [[nodiscard]] const std::vector<TensorDesc>& ioDesc() const noexcept { return m_ioDesc; }

    //! The configuration the model actually uses (= the workerCfg passed to init,
    //! with the batch corrected to the engine's capability). Note this is "the
    //! copy fed to the model", not the one the caller passed in: the getXxx reads
    //! during the model's init leave readKeys traces on it, so logConfigBox must
    //! use it to judge correctly which keys really took effect.
    [[nodiscard]] const ModelConfig& modelConfig() const noexcept { return m_cfg; }

private:
    struct Worker
    {
        std::unique_ptr<IModel> model;   //!< exclusively owned IModel instance
        std::thread thread;              //!< exclusively owned thread
    };

    void workerLoop(std::size_t workerIndex);
    BatchResult runBatch(std::size_t workerIndex, Batch batch);

    ModelConfig m_cfg;               //!< the configuration the model actually uses (after the engine's correction)
    Factory m_factory;
    std::size_t m_maxQueueSize;
    int m_resolvedBatch = 1;         //!< batch after correction to the engine's real capability
    std::vector<TensorDesc> m_ioDesc;    //!< worker[0]'s I/O description (snapshotted during construction)
    std::vector<Worker> m_workers;

    std::queue<std::pair<Batch, std::promise<BatchResult>>> m_tasks;

    mutable std::mutex m_mutex;
    std::condition_variable m_cvTask;      // a task arrived
    std::condition_variable m_cvNotFull;   // the queue has room
    std::condition_variable m_cvIdle;      // everything idle
    std::condition_variable m_cvShutdown;  // shutdown completed (concurrent callers wait on it)
    std::size_t m_activeCount = 0;
    bool m_stop = false;
    bool m_shutdownDone = false;
};

}  // namespace trt_alpha::core
