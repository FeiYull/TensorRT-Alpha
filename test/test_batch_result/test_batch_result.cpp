// =============================================================================
//  test/test_batch_result/test_batch_result.cpp
// -----------------------------------------------------------------------------
//  BatchResult 测试：
//    [1] 默认构造（empty / size）
//    [2] 源信息字段（sourceId / firstFrameIndex）
//    [3] 各任务结果字段独立
//    [4] buffer / views 生命周期
// =============================================================================
#include "trt_alpha/cls/types.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/det/types.hpp"
#include "trt_alpha/seg/types.hpp"

#include <cstdint>
#include <iostream>

using trt_alpha::cls::ClassScore;
using trt_alpha::core::BatchResult;
using trt_alpha::core::Buffer;
using trt_alpha::core::BufferView;
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
    std::cout << "=== BatchResult tests ===\n";

    // [1] 默认构造
    {
        BatchResult r;
        check(r.empty(),          "[1] default empty() == true");
        check(r.size() == 0,      "[1] default size == 0");
        check(r.sourceId == -1,   "[1] default sourceId == -1");
        check(r.firstFrameIndex == 0, "[1] default firstFrameIndex == 0");
        check(r.validCount == 0,  "[1] default validCount == 0");
        check(r.detections.empty(),        "[1] detections empty");
        check(r.segmentations.empty(),     "[1] segmentations empty");
        check(r.classifications.empty(),   "[1] classifications empty");
        check(r.inferenceMs == 0.0,        "[1] inferenceMs == 0");
    }

    // [2] 源信息字段
    {
        BatchResult r;
        r.sourceId = 7;
        r.firstFrameIndex = 100;
        check(r.sourceId == 7,          "[2] sourceId set");
        check(r.firstFrameIndex == 100, "[2] firstFrameIndex set");
    }

    // [3] 各任务结果独立
    {
        BatchResult r;
        r.detections.resize(2);          // 2 张图
        r.segmentations.resize(2);
        r.classifications.resize(2);

        // 图 0：2 个框
        r.detections[0].push_back(Detection{1.f, 2.f, 3.f, 4.f, 0.9f, 0});
        r.detections[0].push_back(Detection{5.f, 6.f, 7.f, 8.f, 0.8f, 1});

        // 图 1：1 个框
        r.detections[1].push_back(Detection{10.f, 20.f, 30.f, 40.f, 0.7f, 2});

        // 图 0：1 个分类
        r.classifications[0].push_back(ClassScore{5, 0.99f});

        check(r.detections[0].size() == 2, "[3] detections[0] size 2");
        check(r.detections[1].size() == 1, "[3] detections[1] size 1");
        check(r.segmentations[0].empty(),  "[3] segmentations[0] empty");
        check(r.classifications[0].size() == 1, "[3] classifications[0] size 1");
        check(r.classifications[1].empty(),     "[3] classifications[1] empty");
    }

    // [4] buffer / views 生命周期
    {
        constexpr int W = 64, H = 64, C = 3;
        auto buf = Buffer::createHost(W, H, C, DataType::UInt8);

        BatchResult r;
        r.buffer = buf;
        r.views.resize(1);
        r.views[0].data = buf->data();
        r.views[0].width = W;
        r.views[0].height = H;
        r.views[0].stride = W * C;
        r.views[0].channels = C;
        r.views[0].dtype = DataType::UInt8;
        r.views[0].space = MemorySpace::Host;
        r.validCount = 1;

        check(!r.empty(), "[4] non-empty");
        check(r.size() == 1, "[4] size == 1");

        // 让局部 shared_ptr 析构，r.buffer 仍持有
        buf.reset();
        check(r.buffer.use_count() == 1, "[4] after reset, use_count == 1");
        check(r.views[0].data != nullptr, "[4] view data still valid");
    }

    std::cout << "=========================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}