// =============================================================================
//  test/test_task_types/test_task_types.cpp
// -----------------------------------------------------------------------------
//  任务类型测试：
//    [1] Detection 默认值 / 赋值
//    [2] Segmentation mask 生命周期（maskOwner 引用计数）
//    [3] ClassScore 默认值
// =============================================================================
#include "trt_alpha/cls/types.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/det/types.hpp"
#include "trt_alpha/seg/types.hpp"

#include <cstdint>
#include <iostream>

using trt_alpha::cls::ClassScore;
using trt_alpha::core::Buffer;
using trt_alpha::core::DataType;
using trt_alpha::core::MemorySpace;
using trt_alpha::det::Detection;
using trt_alpha::seg::Segmentation;

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
    std::cout << "=== Task types tests ===\n";

    // [1] Detection
    {
        Detection d;
        check(d.left == 0.f && d.top == 0.f && d.right == 0.f && d.bottom == 0.f,
              "[1] default coords all 0");
        check(d.confidence == 0.f, "[1] default confidence 0");
        check(d.label == -1,       "[1] default label -1");

        d.left = 10.f; d.top = 20.f; d.right = 100.f; d.bottom = 200.f;
        d.confidence = 0.85f; d.label = 3;
        check(d.left == 10.f && d.right == 100.f, "[1] assign coords");
        check(d.confidence == 0.85f, "[1] assign conf");
        check(d.label == 3,         "[1] assign label");
    }

    // [2] Segmentation 生命周期
    {
        // 构造一个 Segmentation，用 shared_ptr 保证 mask 有效
        Segmentation s;
        s.box.left = 10.f;
        s.box.top = 20.f;

        constexpr int W = 64, H = 64, C = 1;
        auto maskBuf = Buffer::createHost(W, H, C, DataType::UInt8);
        s.mask.data = maskBuf->data();
        s.mask.width = W;
        s.mask.height = H;
        s.mask.stride = W * C;
        s.mask.channels = C;
        s.mask.dtype = DataType::UInt8;
        s.mask.space = MemorySpace::Host;
        s.maskOwner = maskBuf;

        check(s.maskOwner != nullptr,             "[2] maskOwner non-null");
        check(s.maskOwner.use_count() >= 2,       "[2] use_count >= 2 (s + local)");
        check(s.mask.data == maskBuf->data(),     "[2] mask.data == buffer.data");

        // 让局部 shared_ptr 析构，maskOwner 仍持有，data 仍有效
        maskBuf.reset();
        check(s.maskOwner.use_count() == 1,       "[2] after reset, use_count == 1");
        check(s.mask.data != nullptr,             "[2] data still valid (owner alive)");
    }

    // [3] ClassScore
    {
        ClassScore c;
        check(c.label == -1, "[3] default label -1");
        check(c.score == 0.f, "[3] default score 0");

        c.label = 5;
        c.score = 0.99f;
        check(c.label == 5 && c.score == 0.99f, "[3] assign");
    }

    std::cout << "=======================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}