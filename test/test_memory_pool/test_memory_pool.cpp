// =============================================================================
//  test/test_memory_pool/test_memory_pool.cpp
// -----------------------------------------------------------------------------
//  MemoryPool 测试：
//    [1] 缓存命中复用（同尺寸释放后再取拿到同一地址）
//    [2] 数据完整性（Pinned -> Device -> Pinned 往返逐字节比对）
//    [3] 并发安全（4 线程随机尺寸并发申请 / 归还）
//    [4] DeviceBuffer 容量复用（容量够时返回同一地址）
//    [5] 统计准确
//    [6] logAllocBox 可调用（人眼检查输出）
//
//  退出码：全部 PASS 返回 0，任一 FAIL 返回 1。
// =============================================================================
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/memory_pool.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using trt_alpha::core::CudaStream;
using trt_alpha::core::DataType;
using trt_alpha::core::DeviceBuffer;
using trt_alpha::core::MemoryPool;
using trt_alpha::core::MemorySpace;
using trt_alpha::core::PinnedBuffer;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

}  // namespace

int main()
{
    std::cout << "=== MemoryPool tests ===\n";

    const auto pool = MemoryPool::instance();

    // ---------------------------------------------------------------
    // [1] 缓存命中复用
    // ---------------------------------------------------------------
    {
        void* firstPtr = nullptr;
        {
            DeviceBuffer buffer(1ULL << 20);
            firstPtr = buffer.data();
        }
        DeviceBuffer buffer(1ULL << 20);
        const bool hit = (buffer.data() == firstPtr && firstPtr != nullptr);
        check(hit, "[1] cache hit reuse (same pointer)");
    }

    // ---------------------------------------------------------------
    // [2] 数据完整性：Pinned -> Device -> Pinned 往返
    // ---------------------------------------------------------------
    {
        constexpr std::size_t kBytes = 4ULL << 20;
        PinnedBuffer src(kBytes);
        PinnedBuffer dst(kBytes);
        DeviceBuffer device(kBytes);

        auto* s = static_cast<unsigned char*>(src.data());
        auto* d = static_cast<unsigned char*>(dst.data());
        for (std::size_t i = 0; i < kBytes; ++i)
        {
            s[i] = static_cast<unsigned char>(i & 0xFF);
        }
        std::memset(d, 0, kBytes);

        CudaStream stream;
        cudaMemcpyAsync(device.data(), s, kBytes, cudaMemcpyHostToDevice, stream.get());
        cudaMemcpyAsync(d, device.data(), kBytes, cudaMemcpyDeviceToHost, stream.get());
        stream.synchronize();

        const bool ok = (std::memcmp(s, d, kBytes) == 0);
        check(ok, "[2] H2D / D2H integrity (byte-for-byte)");
    }

    // ---------------------------------------------------------------
    // [3] 并发安全：4 线程随机尺寸
    // ---------------------------------------------------------------
    {
        std::atomic<int> failures{0};
        constexpr static int kThreads = 4;
        constexpr static int kIters = 200;

        const auto worker = [&failures](unsigned seed) {
            std::mt19937 rng(seed);
            std::uniform_int_distribution<std::size_t> sizeDist(64ULL << 10, 1ULL << 20);
            for (int i = 0; i < kIters; ++i)
            {
                try
                {
                    const std::size_t bytes = sizeDist(rng);
                    DeviceBuffer device(bytes);
                    PinnedBuffer host(bytes);
                    auto* h = static_cast<unsigned char*>(host.data());
                    h[0] = static_cast<unsigned char>(i);
                    h[bytes - 1] = static_cast<unsigned char>(i >> 8);

                    CudaStream stream;
                    cudaMemcpyAsync(device.data(), h, bytes,
                                    cudaMemcpyHostToDevice, stream.get());
                    stream.synchronize();
                    std::memset(h, 0, bytes);
                    cudaMemcpyAsync(h, device.data(), bytes,
                                    cudaMemcpyDeviceToHost, stream.get());
                    stream.synchronize();

                    if (h[0] != static_cast<unsigned char>(i) ||
                        h[bytes - 1] != static_cast<unsigned char>(i >> 8))
                    {
                        ++failures;
                    }
                }
                catch (...)
                {
                    ++failures;
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t)
        {
            threads.emplace_back(worker, 1234u + static_cast<unsigned>(t));
        }
        for (auto& th : threads) { th.join(); }

        check(failures.load() == 0, "[3] 4-thread alloc/free stress (data OK)");
    }

    // ---------------------------------------------------------------
    // [4] DeviceBuffer 容量复用
    // ---------------------------------------------------------------
    {
        DeviceBuffer b(1ULL << 20);
        void* p1 = b.data();
        b.allocate(1ULL << 19);   // 更小：容量够，不重分配
        check(b.data() == p1, "[4] allocate(smaller) keeps same ptr");
        check(b.bytes() >= (1ULL << 20), "[4] capacity unchanged");
    }

    // ---------------------------------------------------------------
    // [5] 统计准确
    // ---------------------------------------------------------------
    {
        auto stats = pool->stats(MemoryPool::Kind::Device);
        std::cout << "       device stats: requests=" << stats.requests
                  << " hits=" << stats.hits
                  << " cudaAllocs=" << stats.cudaAllocs
                  << " inUse=" << (stats.inUseBytes >> 10) << " KiB"
                  << " cached=" << (stats.cachedBytes >> 10) << " KiB"
                  << " peak=" << (stats.peakInUseBytes >> 10) << " KiB\n";
        check(stats.requests > 0, "[5] device stats non-zero requests");
        check(stats.peakInUseBytes > 0, "[5] device peak > 0");
    }

    // ---------------------------------------------------------------
    // [6] logAllocBox（人眼检查输出）
    // ---------------------------------------------------------------
    {
        trt_alpha::core::detail::AllocInfo info;
        info.name = "input_nchw";
        info.batch = 4;
        info.channels = 3;
        info.height = 640;
        info.width = 640;
        info.dtype = DataType::Float32;
        info.bytes = static_cast<std::size_t>(4) * 3 * 640 * 640 * 4;
        info.space = MemorySpace::Device;
        trt_alpha::core::detail::logAllocBox(info);
        check(true, "[6] logAllocBox called (check output above)");
    }

    std::cout << "========================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}