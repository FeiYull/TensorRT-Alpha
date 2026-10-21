// =============================================================================
//  trt_alpha :: core :: gpu_event
// -----------------------------------------------------------------------------
//  GpuEvent -- a one-shot "the copy I queued has landed" marker.
//
//  Why this exists:
//    A data source that decodes on the GPU produces the pixel block on the card.
//    The host copy that the renderer needs is filled with an async memcpy. If the
//    producer waited for that copy itself, the wait would sit in the source
//    thread -- which is the bottleneck on a real decode pipeline -- even though
//    nothing downstream is ready to look at the pixels yet. Instead the producer
//    records an event and hands it over with the batch; whoever touches the host
//    pixels waits there, by which time the copy has long finished.
//
//  Why the header does not mention cudaEvent_t:
//    core's public headers stay self-contained and CUDA-free where practical
//    (see cuda_stream.hpp for the one exception). The handle lives in the impl.
//
//  Semantics:
//    * record() marks the legacy default stream. That is deliberate: the async
//      copies in the sources are queued on that stream, and a blocking stream
//      (CudaStream) is implicitly ordered against it.
//    * wait() is idempotent and may be called from any thread, any number of
//      times, including concurrently: cudaEventSynchronize is thread-safe.
// =============================================================================
#pragma once

#include <memory>

namespace trt_alpha::core {

class GpuEvent
{
public:
    GpuEvent();
    ~GpuEvent();

    GpuEvent(const GpuEvent&) = delete;
    GpuEvent& operator=(const GpuEvent&) = delete;
    GpuEvent(GpuEvent&&) = delete;
    GpuEvent& operator=(GpuEvent&&) = delete;

    //! Mark everything already queued on the legacy default stream.
    void record();

    //! Block until the marked point has been reached.
    void wait() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace trt_alpha::core
