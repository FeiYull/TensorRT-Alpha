// =============================================================================
//  test/test_batch/test_batch.cpp
// -----------------------------------------------------------------------------
//  Batch 测试（用 Buffer / BufferView）：
//    [1] 默认构造 + validate 失败
//    [2] 正常构造 + validate 通过
//    [3] validate 失败路径
//    [4] 拷贝语义（views 独立，buffer 共享）
//    [5] empty / size
// =============================================================================
#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using trt_alpha::core::Batch;
using trt_alpha::core::Buffer;
using trt_alpha::core::BufferView;
using trt_alpha::core::DataType;
using trt_alpha::core::MemorySpace;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

Batch makeTestBatch(int batchSize, int validCount)
{
    constexpr int W = 64, H = 64, C = 3;

    // 分配一整块：W*batchSize 宽、H 高、3 通道的 uint8
    auto buffer = Buffer::createHost(W * batchSize, H, C, DataType::UInt8);

    Batch b;
    b.sourceId = 0;
    b.firstFrameIndex = 0;
    b.buffer = buffer;
    b.validCount = validCount;

    const int oneFrameBytes = W * H * C;
    for (int i = 0; i < batchSize; ++i)
    {
        BufferView v;
        v.data = buffer->data() + static_cast<std::size_t>(i) * oneFrameBytes;
        v.width = W;
        v.height = H;
        v.stride = W * C;
        v.channels = C;
        v.dtype = DataType::UInt8;
        v.space = MemorySpace::Host;
        b.views.push_back(v);
    }
    return b;
}

}  // namespace

int main()
{
    std::cout << "=== Batch tests ===\n";

    // [1] 默认
    {
        Batch b;
        check(b.empty(), "[1] default empty");
        check(b.size() == 0, "[1] default size 0");
        check(b.sourceId == -1, "[1] default sourceId");
        check(b.firstFrameIndex == 0, "[1] default firstFrameIndex");
        std::string err;
        check(!b.validate(&err), "[1] default validate fails");
    }

    // [2] 正常
    {
        Batch b = makeTestBatch(4, 4);
        check(!b.empty(), "[2] non-empty");
        check(b.size() == 4, "[2] size 4");
        check(b.validCount == 4, "[2] validCount 4");
        std::string err;
        check(b.validate(&err), "[2] validate passes");
    }

    // [3] validate 失败
    {
        Batch b = makeTestBatch(4, 4);
        b.validCount = 5;
        std::string err;
        check(!b.validate(&err), "[3a] validCount > size fails");

        b = makeTestBatch(4, 4);
        b.views[2].width = 128;
        check(!b.validate(&err), "[3b] inconsistent dims fails");

        b = makeTestBatch(4, 4);
        b.views[1].data = nullptr;
        check(!b.validate(&err), "[3c] nullptr view data fails");
    }

    // [4] 拷贝语义
    {
        Batch b1 = makeTestBatch(4, 4);
        Batch b2 = b1;

        check(b2.size() == b1.size(), "[4] size equal");
        check(b2.buffer == b1.buffer, "[4] buffer shared");
        check(b2.views.size() == b1.views.size(), "[4] views size equal");

        b2.views[0].width = 999;
        check(b1.views[0].width == 64, "[4] views array independent");
    }

    // [5] empty / size
    {
        Batch b;
        check(b.empty(), "[5] empty default");

        b.buffer = Buffer::createHost(64, 64, 3, DataType::UInt8);
        check(b.empty(), "[5] still empty (no views)");

        b.views.push_back(BufferView{});
        check(!b.empty(), "[5] not empty");
        check(b.size() == 1, "[5] size 1");
    }

    std::cout << "===================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}