// =============================================================================
//  trt_alpha :: core :: batch_stage（实现）
// =============================================================================
#include "trt_alpha/core/batch_stage.hpp"

#include "trt_alpha/core/logger.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace trt_alpha::core {
namespace {

void checkCuda(cudaError_t err, const char* who, const char* op)
{
    if (err != cudaSuccess)
    {
        const std::string msg = std::string(who) + ": " + op + " failed: " +
                                cudaGetErrorString(err);
        TRT_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }
}

}  // namespace

const void* stageBatchPixels(const Batch& batch, DeviceBuffer& staging,
                             cudaStream_t stream, const char* who)
{
    // ---- fast path: the pixels are already where the model wants them -------
    if (batch.deviceBuffer != nullptr)
    {
        // Two shapes arrive here:
        //   * host copy for the renderer + device copy for the model (`views`
        //     describes the host block), and
        //   * device only (`needHostCopy == false`; `views` describes the device
        //     block, same geometry, `space == Device`).
        // Both skip the upload, so the only useful check is the one that keeps a
        // later flat memcpy honest -- and it needs geometry, which a batch with no
        // views does not have.
        if (!batch.views.empty())
        {
            const BufferView& first = batch.views[0];
            const std::size_t tightStride = static_cast<std::size_t>(first.width) * 3u;
            if (first.stride != static_cast<int>(tightStride))
            {
                throw std::runtime_error(std::string(who) + ": source stride is " +
                                         std::to_string(first.stride) +
                                         " bytes, expected a tight " +
                                         std::to_string(tightStride));
            }
            const std::size_t total = tightStride *
                                      static_cast<std::size_t>(first.height) *
                                      batch.views.size();
            if (batch.deviceBuffer->byteSize() < total)
            {
                throw std::runtime_error(std::string(who) + ": device buffer holds " +
                                         std::to_string(batch.deviceBuffer->byteSize()) +
                                         " bytes but the batch needs " +
                                         std::to_string(total));
            }
            TRT_LOG_DEBUG(who << ": pixels already on the device ("
                         << total << " bytes), H2D skipped");
        }
        return batch.deviceBuffer->data();
    }

    // ---- classic path: host pixels, upload them ----------------------------
    if (batch.buffer == nullptr || batch.views.empty())
    {
        throw std::runtime_error(std::string(who) + ": batch has no pixels");
    }

    const BufferView& first = batch.views[0];
    const std::size_t oneImage = static_cast<std::size_t>(first.width) *
                                 static_cast<std::size_t>(first.height) * 3u;
    const std::size_t total = oneImage * batch.views.size();

    // The upload below is a single flat memcpy of the whole batch, which is only
    // correct while the source block is tightly packed. A padded source would
    // silently skew every frame after the first, so reject it instead.
    const std::size_t tightStride = static_cast<std::size_t>(first.width) * 3u;
    if (first.stride != static_cast<int>(tightStride))
    {
        throw std::runtime_error(std::string(who) + ": source stride is " +
                                 std::to_string(first.stride) +
                                 " bytes, expected a tight " +
                                 std::to_string(tightStride));
    }

    if (staging.bytes() < total)
    {
        // 源图尺寸变化触发扩容（少见）
        staging.allocate(total);
    }

    // The host pixels may still be in flight: a source that queues its copy and
    // hands the batch straight over marks them asynchronously. The pipeline waits
    // before calling here, but this is the function that actually reads them, so
    // it must not depend on that. Idempotent and ~free once the copy has landed.
    if (batch.hostReady)
    {
        batch.hostReady->wait();
    }

    checkCuda(cudaMemcpyAsync(staging.data(), batch.buffer->data(), total,
                              cudaMemcpyHostToDevice, stream),
              who, "cudaMemcpyAsync(H2D)");
    checkCuda(cudaStreamSynchronize(stream), who, "cudaStreamSynchronize");

    TRT_LOG_DEBUG(who << ": H2D done, " << total << " bytes");
    return staging.data();
}

}  // namespace trt_alpha::core
