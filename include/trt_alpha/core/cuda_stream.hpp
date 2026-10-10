// =============================================================================
//  trt_alpha :: core :: cuda_stream
// -----------------------------------------------------------------------------
//  CudaStream -- an RAII wrapper around cudaStream_t.
//
//  Design:
//    * Created with the default flag (blocking stream): it synchronizes
//      automatically with the legacy default stream, so it is timing-safe
//      against any external code running on the default stream (e.g. the OpenCV
//      CUDA backend).
//    * Copy and move are disabled: a stream is an exclusive resource handle and
//      moving it would introduce ownership ambiguity.
//    * Not tied to MemoryPool -- the pool only allocates memory, it does not
//      care which stream that memory is used on. The stream/memory relationship
//      is maintained by the user ("I use buffer B on stream S and must not free
//      B until I am done").
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace trt_alpha::core {

class CudaStream
{
public:
    CudaStream();
    ~CudaStream();

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&&) = delete;
    CudaStream& operator=(CudaStream&&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return m_stream; }

    //! Synchronize every operation enqueued on this stream.
    void synchronize();

private:
    cudaStream_t m_stream = nullptr;
};

}  // namespace trt_alpha::core
