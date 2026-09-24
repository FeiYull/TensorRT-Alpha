// =============================================================================
//  test/test_buffer_view/test_buffer_view.cpp
// -----------------------------------------------------------------------------
//  BufferView 测试：
//    [1] 默认构造
//    [2] 正常构造（BGR8 图像） + byteSize / tightStride
//    [3] 空视图多种情况
//    [4] Device 空间
//    [5] 不同类型（fp32）的 stride 计算
// =============================================================================
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

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

}  // namespace

int main()
{
    std::cout << "=== BufferView tests ===\n";

    // [1] 默认构造
    {
        BufferView v;
        check(v.data == nullptr,             "[1] default data == nullptr");
        check(v.dtype == DataType::UInt8,    "[1] default dtype == UInt8");
        check(v.space == MemorySpace::Host,  "[1] default space == Host");
        check(v.empty(),                     "[1] default empty() == true");
        check(v.byteSize() == 0,             "[1] default byteSize == 0");
    }

    // [2] BGR8 图像
    {
        constexpr int W = 640, H = 480, C = 3;
        std::vector<std::uint8_t> buf(W * H * C, 128);

        BufferView v;
        v.data = buf.data();
        v.width = W;
        v.height = H;
        v.channels = C;
        v.stride = W * C;
        v.dtype = DataType::UInt8;
        v.space = MemorySpace::Host;

        check(!v.empty(),                                     "[2] non-empty");
        check(v.byteSize() == static_cast<std::size_t>(W) * H * C, "[2] byteSize");
        check(v.tightStride() == static_cast<std::size_t>(W) * C, "[2] tightStride");
    }

    // [3] 空视图
    {
        BufferView v1;
        check(v1.empty(), "[3] nullptr empty");

        std::vector<std::uint8_t> buf(16, 0);
        BufferView v2;
        v2.data = buf.data();
        v2.width = 0; v2.height = 4; v2.channels = 3;
        check(v2.empty(), "[3] width==0 empty");

        BufferView v3;
        v3.data = buf.data();
        v3.width = 4; v3.height = 4; v3.channels = 0;
        check(v3.empty(), "[3] channels==0 empty");
    }

    // [4] Device
    {
        BufferView v;
        v.data = reinterpret_cast<const std::uint8_t*>(0x1000);
        v.width = 64; v.height = 64; v.channels = 3;
        v.stride = 64 * 3;
        v.dtype = DataType::UInt8;
        v.space = MemorySpace::Device;
        check(!v.empty(), "[4] device non-empty");
        check(v.space == MemorySpace::Device, "[4] space == Device");
    }

    // [5] fp32 张量（NCHW 中单个通道 H×W）
    {
        constexpr int W = 320, H = 320, C = 1;
        constexpr int elementBytes = 4;  // fp32
        BufferView v;
        v.data = reinterpret_cast<const std::uint8_t*>(0x2000);
        v.width = W; v.height = H; v.channels = C;
        v.stride = W * C * elementBytes;
        v.dtype = DataType::Float32;
        check(v.tightStride() == static_cast<std::size_t>(W) * C * elementBytes,
              "[5] fp32 tightStride uses 4 bytes/elem");
        check(v.byteSize() == static_cast<std::size_t>(W) * H * C * elementBytes,
              "[5] fp32 byteSize");
    }

    std::cout << "========================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}