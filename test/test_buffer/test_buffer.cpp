// =============================================================================
//  test/test_buffer/test_buffer.cpp
// -----------------------------------------------------------------------------
//  Buffer 测试：
//    [1] createHost（uint8）
//    [2] createHost（float32）
//    [3] fromHostData 逐行拷贝（srcStride > tight）
//    [4] createDevice
//    [5] view()
//    [6] 尺寸非法抛异常
// =============================================================================
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

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

}  // namespace

int main()
{
    std::cout << "=== Buffer tests ===\n";

    // [1] uint8 图像
    {
        auto buf = Buffer::createHost(640, 480, 3, DataType::UInt8);
        check(buf != nullptr,            "[1] non-null");
        check(buf->width() == 640,       "[1] width");
        check(buf->height() == 480,      "[1] height");
        check(buf->channels() == 3,      "[1] channels");
        check(buf->stride() == 640 * 3,  "[1] stride");
        check(buf->dtype() == DataType::UInt8, "[1] dtype UInt8");
        check(buf->space() == MemorySpace::Host, "[1] space Host");
        check(buf->byteSize() == static_cast<std::size_t>(640) * 480 * 3,
              "[1] byteSize");
    }

    // [2] float32 张量
    {
        auto buf = Buffer::createHost(320, 320, 1, DataType::Float32);
        check(buf->stride() == 320 * 4,   "[2] stride = 320 * 4");
        check(buf->dtype() == DataType::Float32, "[2] dtype Float32");
        check(buf->byteSize() == static_cast<std::size_t>(320) * 320 * 4,
              "[2] byteSize accounts for 4 bytes/elem");
    }

    // [3] fromHostData（srcStride > tight）
    {
        constexpr int W = 4, H = 2, C = 3;
        constexpr int SRC_STRIDE = W * C + 5;

        std::vector<std::uint8_t> src(H * SRC_STRIDE, 0xFF);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W * C; ++x)
                src[y * SRC_STRIDE + x] = static_cast<std::uint8_t>(y * 100 + x);

        auto buf = Buffer::fromHostData(src.data(), W, H, C, SRC_STRIDE, DataType::UInt8);
        check(buf->stride() == W * C, "[3] dst stride compact");
        check(buf->byteSize() == static_cast<std::size_t>(W) * H * C, "[3] dst byteSize");

        bool ok = true;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W * C; ++x)
                if (buf->data()[y * W * C + x] != static_cast<std::uint8_t>(y * 100 + x))
                    ok = false;
        check(ok, "[3] padding stripped");
    }

    // [4] Device
    {
        try
        {
            auto buf = Buffer::createDevice(64, 64, 3, DataType::UInt8);
            check(buf->space() == MemorySpace::Device, "[4] space Device");
            check(buf->stride() == 64 * 3, "[4] stride");
        }
        catch (const std::exception& e)
        {
            std::cout << "[WARN] [4] createDevice threw: " << e.what() << "\n";
        }
    }

    // [5] view()
    {
        auto buf = Buffer::createHost(100, 50, 3, DataType::UInt8);
        BufferView v = buf->view();
        check(v.data == buf->data(), "[5] view.data");
        check(v.width == 100 && v.height == 50, "[5] view dims");
        check(v.stride == 100 * 3, "[5] view stride");
        check(v.dtype == DataType::UInt8, "[5] view dtype");
        check(v.space == MemorySpace::Host, "[5] view space");
    }

    // [6] 非法尺寸
    {
        bool threw = false;
        try { (void)Buffer::createHost(0, 100, 3, DataType::UInt8); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[6] zero width throws");

        threw = false;
        try { (void)Buffer::fromHostData(nullptr, 10, 10, 3, 30, DataType::UInt8); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[6] nullptr srcData throws");
    }

    std::cout << "======================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}