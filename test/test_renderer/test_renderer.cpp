// =============================================================================
//  test/test_renderer/test_renderer.cpp
// -----------------------------------------------------------------------------
//  OpenCVRenderer 测试（不依赖 engine，手工造数据）：
//    [1] drawResult：画框后，对应像素颜色变了
//    [2] save：存盘后文件存在
//    [3] show：调用不崩（不检查窗口）
//    [4] 空 result：不崩
// =============================================================================
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/core/class_info.hpp"
#include "trt_alpha/det/types.hpp"
#include "trt_alpha/renderer/opencv_renderer.hpp"

#include <opencv2/highgui.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace fs = std::filesystem;

using trt_alpha::core::BatchResult;
using trt_alpha::core::Buffer;
using trt_alpha::core::BufferView;
using trt_alpha::core::ClassInfo;
using trt_alpha::core::DataType;
using trt_alpha::core::MemorySpace;
using trt_alpha::det::Detection;
using trt_alpha::renderer::OpenCVRenderer;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

//! 造一个 100x100 的纯色 BatchResult（1 张图，2 个框）。
BatchResult makeTestResult(int width, int height)
{
    auto buffer = Buffer::createHost(width, height, 3, DataType::UInt8);
    // 填灰
    std::memset(buffer->mutableData(), 128, buffer->byteSize());

    BatchResult r;
    r.buffer = buffer;
    r.validCount = 1;
    r.firstFrameIndex = 0;

    BufferView v;
    v.data = buffer->data();
    v.width = width;
    v.height = height;
    v.stride = width * 3;
    v.channels = 3;
    v.dtype = DataType::UInt8;
    v.space = MemorySpace::Host;
    r.views.push_back(v);

    r.detections.resize(1);
    Detection d1; d1.left = 10; d1.top = 10; d1.right = 50; d1.bottom = 50;
    d1.confidence = 0.9f; d1.label = 0;
    Detection d2; d2.left = 60; d2.top = 60; d2.right = 90; d2.bottom = 90;
    d2.confidence = 0.8f; d2.label = 1;
    r.detections[0].push_back(d1);
    r.detections[0].push_back(d2);

    return r;
}

}  // namespace

int main()
{
    std::cout << "=== OpenCVRenderer tests ===\n";

    OpenCVRenderer renderer;

    std::vector<ClassInfo> classNames = {
        ClassInfo{"person", 255, 0, 0},
        ClassInfo{"bicycle", 0, 255, 0},
    };

    // ---------------------------------------------------------------
    // [1] drawResult 改了像素
    // ---------------------------------------------------------------
    {
        BatchResult r = makeTestResult(100, 100);
        const std::uint8_t* before = r.views[0].data;

        // 记录 (10, 10) 位置的原始颜色（灰 = 128, 128, 128）
        const std::uint8_t orig0 = before[10 * 100 * 3 + 10 * 3 + 0];
        check(orig0 == 128, "[1] before draw: pixel is gray (128)");

        renderer.drawResult(r, classNames);

        // (10, 10) 是框的左上角，(10, 10) 应该被画了框线（颜色 = colorForLabel(0)）
        const std::uint8_t after0 = before[10 * 100 * 3 + 10 * 3 + 0];
        check(after0 != 128, "[1] after draw: pixel at (10,10) changed");
    }

    // ---------------------------------------------------------------
    // [2] save 存盘
    // ---------------------------------------------------------------
    {
        BatchResult r = makeTestResult(100, 100);
        renderer.drawResult(r, classNames);

        const std::string outDir = "test_renderer_out";
        std::error_code ec;
        fs::remove_all(outDir, ec);   // 清掉旧文件

        renderer.save(r, outDir, "img_");

        const fs::path expected = fs::path(outDir) / "img_0.jpg";
        check(fs::exists(expected), "[2] saved file exists (img_0.jpg)");

        // 清理
        fs::remove_all(outDir, ec);
    }

    // ---------------------------------------------------------------
    // [3] show 调用不崩
    // ---------------------------------------------------------------
    {
        BatchResult r = makeTestResult(100, 100);
        renderer.drawResult(r, classNames);
        renderer.show(r, "test_renderer_window");
        cv::destroyAllWindows();
        check(true, "[3] show() called without crash");
    }

    // ---------------------------------------------------------------
    // [4] 空 result 不崩
    // ---------------------------------------------------------------
    {
        BatchResult empty;
        renderer.drawResult(empty, classNames);
        renderer.save(empty, "test_renderer_out_empty", "x_");
        renderer.show(empty, "test_renderer_empty");
        check(true, "[4] empty result handled without crash");

        std::error_code ec;
        fs::remove_all("test_renderer_out_empty", ec);
    }

    std::cout << "============================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}