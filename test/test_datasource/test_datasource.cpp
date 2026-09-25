// =============================================================================
//  test/test_datasource/test_datasource.cpp
// -----------------------------------------------------------------------------
//  OpenCVSource 测试：
//    [1] 图片模式：读一张图 → Batch{validCount=1}
//    [2] 图片目录模式：扫目录 → 读多张 → 攒批
//    [3] batchSize=4 但只有 2 张图 → 第二个 batch validCount=2
//    [4] 视频模式：读前 N 帧
//    [5] 构造失败（文件不存在）抛异常
//
//  用法：
//    test_datasource                          # 用 data/bus.jpg
//    test_datasource <image_path>             # 指定图片
// =============================================================================
#include "trt_alpha/datasource/opencv_source.hpp"
#include "trt_alpha/datasource/source_config.hpp"
#include "trt_alpha/core/paths.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using trt_alpha::core::Batch;
using trt_alpha::datasource::OpenCVSource;
using trt_alpha::datasource::SourceConfig;
using trt_alpha::datasource::SourceType;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

}  // namespace

int main(int argc, char** argv)
{
    std::cout << "=== OpenCVSource tests ===\n";

    // 默认图片路径（相对项目根）
    std::string imagePath = "data/bus.jpg";
    if (argc >= 2)
    {
        imagePath = argv[1];
    }

    // ---------------------------------------------------------------
    // [1] 图片模式
    // ---------------------------------------------------------------
    {
        try
        {
            SourceConfig cfg;
            cfg.type = SourceType::Image;
            cfg.path = imagePath;
            cfg.batchSize = 1;
            cfg.sourceId = 0;

            OpenCVSource source(cfg);

            Batch batch;
            const bool ok = source.next(batch);

            check(ok, "[1] next() returned true");
            check(batch.validCount == 1, "[1] validCount == 1");
            check(batch.views.size() == 1, "[1] views.size() == 1");
            check(batch.views[0].width > 0 && batch.views[0].height > 0,
                  "[1] view has valid dimensions");
            check(batch.views[0].channels == 3, "[1] channels == 3");

            std::cout << "       image size: " << batch.views[0].width << "x"
                      << batch.views[0].height << "\n";

            // 第二次 next 应返回 false（图片模式只读一次）
            Batch batch2;
            const bool ok2 = source.next(batch2);
            check(!ok2, "[1] second next() returns false (single image)");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [1] exception: " << e.what() << "\n";
            ++g_failures;
        }
    }

    // ---------------------------------------------------------------
    // [2] batchSize=4，图片模式 → 第一次 validCount=1，之后结束
    // ---------------------------------------------------------------
    {
        try
        {
            SourceConfig cfg;
            cfg.type = SourceType::Image;
            cfg.path = imagePath;
            cfg.batchSize = 4;

            OpenCVSource source(cfg);

            Batch batch;
            const bool ok = source.next(batch);
            check(ok, "[2] first next() true");
            check(batch.validCount == 1, "[2] validCount == 1 (only 1 image)");
            check(batch.views.size() == 4, "[2] views.size() == 4 (fixed batch)");

            Batch batch2;
            const bool ok2 = source.next(batch2);
            check(!ok2, "[2] second next() false (end of source)");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [2] exception: " << e.what() << "\n";
            ++g_failures;
        }
    }

    // ---------------------------------------------------------------
    // [3] 构造失败（文件不存在）
    // ---------------------------------------------------------------
    {
        bool threw = false;
        try
        {
            SourceConfig cfg;
            cfg.type = SourceType::Image;
            cfg.path = "no_such_file_12345.jpg";
            cfg.batchSize = 1;
            OpenCVSource source(cfg);
        }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[3] nonexistent image throws");
    }

    // ---------------------------------------------------------------
    // [4] requestStop 后 next 立即返回 false
    // ---------------------------------------------------------------
    {
        try
        {
            SourceConfig cfg;
            cfg.type = SourceType::Image;
            cfg.path = imagePath;
            cfg.batchSize = 1;

            OpenCVSource source(cfg);
            source.requestStop();

            Batch batch;
            const bool ok = source.next(batch);
            check(!ok, "[4] next() after requestStop returns false");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [4] exception: " << e.what() << "\n";
            ++g_failures;
        }
    }
    // ---------------------------------------------------------------
    // [5] 视频模式：读前 2 个 batch（batchSize=2）
    // ---------------------------------------------------------------
    {
        try
        {
            SourceConfig cfg;
            cfg.type = SourceType::Video;
            cfg.path = "data/people.mp4";
            cfg.batchSize = 2;

            OpenCVSource source(cfg);

            int batchCount = 0;
            int totalValid = 0;
            for (int i = 0; i < 2; ++i)
            {
                Batch batch;
                const bool ok = source.next(batch);
                if (!ok)
                {
                    break;
                }
                ++batchCount;
                totalValid += batch.validCount;

                std::cout << "       video batch #" << i
                          << " validCount=" << batch.validCount << "/" << 2
                          << " size=" << batch.views[0].width << "x"
                          << batch.views[0].height << "\n";
            }

            check(batchCount == 2, "[5] read 2 video batches");
            check(totalValid == 4, "[5] total valid frames == 4");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [5] exception: " << e.what() << "\n";
            ++g_failures;
        }
    }
    std::cout << "============================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}