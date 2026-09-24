// =============================================================================
//  trt_alpha :: core :: cuda_stream（实现）
// =============================================================================
#include "trt_alpha/core/cuda_stream.hpp"

#include "trt_alpha/core/logger.hpp"

#include <stdexcept>
#include <string>

namespace trt_alpha::core {
namespace {

void checkCuda(cudaError_t err, const char* op)
{
    if (err != cudaSuccess)
    {
        const std::string msg = std::string("CudaStream: ") + op + " failed: " +
                                cudaGetErrorString(err);
        TRT_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }
}

}  // namespace

CudaStream::CudaStream()
{
    checkCuda(cudaStreamCreate(&m_stream), "cudaStreamCreate");
    TRT_LOG_DEBUG("CudaStream: created " << static_cast<const void*>(m_stream));
}

CudaStream::~CudaStream()
{
    if (m_stream != nullptr)
    {
        // 析构 noexcept：忽略返回值（进程退出路径）
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

void CudaStream::synchronize()
{
    checkCuda(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize");
}

}  // namespace trt_alpha::core