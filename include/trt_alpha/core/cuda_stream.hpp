// =============================================================================
//  trt_alpha :: core :: cuda_stream
// -----------------------------------------------------------------------------
//  CudaStream —— cudaStream_t 的 RAII 封装。
//
//  设计：
//    * 用默认 flag（阻塞流）创建：与 legacy default stream 自动互相同步，
//      因此与任何外部跑在默认流上的代码（如 OpenCV CUDA 后端）时序安全。
//    * 禁拷贝、禁移动：流是"独占资源句柄"，移动会引入所有权歧义。
//    * 不绑定到 MemoryPool —— 池只分配内存，不管"内存在哪条流上用"。
//      流和内存的关系由使用者维护（"我在流 S 上用 buffer B，用完前不释放 B"）。
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

    //! 同步本流上所有已入队的操作。
    void synchronize();

private:
    cudaStream_t m_stream = nullptr;
};

}  // namespace trt_alpha::core