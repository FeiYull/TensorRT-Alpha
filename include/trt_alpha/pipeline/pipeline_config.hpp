// =============================================================================
//  trt_alpha :: pipeline :: pipeline_config
// -----------------------------------------------------------------------------
//  PipelineConfig -- configuration for the three-stage pipeline.
//
//  Data flow:
//    [DataSource] -> [InferencePool] -> [ResultQueue] -> [Renderer]
//
//  Multiple pools + routing:
//    * pools[i]        -- the i-th inference pool (different models)
//    * sourceToPool[i] -- which pool source i uses (an index)
//    * when sourceToPool is empty: every source uses pools[0]
//
//  Save / show:
//    * saveEnabled: whether to save to disk
//    * saveDir: output directory (relative to the project root, or absolute)
//    * showEnabled: whether to display
//    * showWindow: window title
// =============================================================================
#pragma once

#include "trt_alpha/core/bounded_queue.hpp"
#include "trt_alpha/core/class_info.hpp"
#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/renderer/i_renderer.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::core {
class InferencePool;
}

namespace trt_alpha::pipeline {

struct PipelineConfig
{
    // ---- Data sources (created by the caller; ownership moves to Pipeline) ----
    std::vector<std::unique_ptr<datasource::IDataSource>> sources;

    // ---- Inference pools (created by the caller; lifetime managed by the caller) ----
    std::vector<core::InferencePool*> pools;

    // ---- Source -> pool routing ----
    //! Its length should equal sources.size().
    //! Empty = every source uses pools[0].
    //! sourceToPool[i] = k means source i uses pools[k].
    std::vector<std::size_t> sourceToPool;

    // ---- Rendering ----
    renderer::IRenderer* renderer = nullptr;
    std::vector<core::ClassInfo> classNames;

    // ---- Result queue ----
    std::size_t resultQueueSize = 32;

    //! Policy when the queue is full. The default is Block -- batch processing
    //! (video files / image directories) requires dropping no frames, so it
    //! would rather make the source thread wait for the renderer to catch up
    //! than silently drop. A real-time source (camera / RTSP stream) should
    //! switch explicitly to DropOldest: there it is better to drop old frames
    //! than to let latency grow without bound.
    core::QueueFullPolicy queueFullPolicy = core::QueueFullPolicy::Block;

    //! Whether to push the "post-render results" on to processedQueue (for
    //! popProcessed to pick up). Default false: nothing should be produced when
    //! nobody consumes, or under the Block policy the render thread would fill
    //! the queue and then block forever. Only callers that really call
    //! popProcessed (e.g. Infer::async) set it true.
    bool exposeProcessed = false;

    // ---- Save / show ----
    bool saveEnabled = false;
    std::string saveDir = "save";
    bool showEnabled = false;
    std::string showWindow = "trt_alpha";

    // ---- Timeout when waiting for sources to stop (ms) ----
    int stopTimeoutMs = 2000;
};

}  // namespace trt_alpha::pipeline
