// =============================================================================
//  trt_alpha :: core :: gpu_event（实现）
// =============================================================================
#include "trt_alpha/core/gpu_event.hpp"

#include "trt_alpha/core/logger.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace trt_alpha::core {

struct GpuEvent::Impl
{
    cudaEvent_t event = nullptr;
};

namespace {

void checkCuda(cudaError_t err, const char* op)
{
    if (err != cudaSuccess)
    {
        const std::string msg = std::string("GpuEvent: ") + op + " failed: " +
                                cudaGetErrorString(err);
        TRT_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }
}

}  // namespace

GpuEvent::GpuEvent()
    : m_impl(std::make_unique<Impl>())
{
    // Timing is disabled: this event is a barrier, not a stopwatch, and the
    // timestamping variant is measurably more expensive to record.
    checkCuda(cudaEventCreateWithFlags(&m_impl->event, cudaEventDisableTiming),
              "cudaEventCreateWithFlags");
}

GpuEvent::~GpuEvent()
{
    if (m_impl != nullptr && m_impl->event != nullptr)
    {
        // 析构 noexcept：忽略返回值（进程退出路径）
        cudaEventDestroy(m_impl->event);
        m_impl->event = nullptr;
    }
}

void GpuEvent::record()
{
    checkCuda(cudaEventRecord(m_impl->event, nullptr), "cudaEventRecord");
}

void GpuEvent::wait() const
{
    if (m_impl == nullptr || m_impl->event == nullptr)
    {
        return;
    }
    checkCuda(cudaEventSynchronize(m_impl->event), "cudaEventSynchronize");
}

}  // namespace trt_alpha::core
