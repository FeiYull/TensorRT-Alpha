// =============================================================================
//  test/test_infer/test_infer.cpp
// -----------------------------------------------------------------------------
//  Infer 高层 API 测试：
//    [1] 同步 run()：单张图
//    [2] 覆盖 conf_thresh = 0.9 → 框数变少
//    [3] frames() / boxes() 一致
//    [4] 异步 async()：视频源
//
//  用法：
//    test_infer                                       # 用默认路径
//    test_infer <image>                               # 指定图片
//    test_infer <image> <video>                       # 指定图片 + 视频
// =============================================================================
#include "trt_alpha/infer/infer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

void step(const char* msg)
{
    std::cout << "[STEP] " << msg << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    std::cout << "=== Infer tests ===\n";

    const std::string imagePath = (argc >= 2) ? argv[1] : "data/bus.jpg";
    const std::string videoPath = (argc >= 3) ? argv[2] : "data/people.mp4";

    std::cout << "image : " << imagePath << "\n";
    std::cout << "video : " << videoPath << "\n";

    try
    {
        // ---------------------------------------------------------------
        // [1] 同步 run()：单张图
        // ---------------------------------------------------------------
        step("1. sync run() on single image");
        {
            trt_alpha::InferParams p;
            p.model_type  = trt_alpha::ModelType::yolov8;
            p.config_path = "configs/yolov8.ini";
            p.source      = imagePath;

            trt_alpha::Infer model(p);
            trt_alpha::Result r = model.run();

            std::cout << "       result: size=" << r.size()
                      << " valid_count=" << r.valid_count()
                      << " inference_ms=" << r.inference_ms() << "\n";

            check(!r.empty(),          "[1] result non-empty");
            check(r.size() == 1,       "[1] size == 1");
            check(r.valid_count() == 1, "[1] valid_count == 1");

            const auto& boxes = r[0].boxes();
            std::cout << "       boxes: " << boxes.size() << "\n";
            check(!boxes.empty(),      "[1] at least 1 detection");

            for (std::size_t i = 0; i < boxes.size() && i < 5; ++i)
            {
                const auto& b = boxes[i];
                std::cout << "         [" << i << "] label=" << b.label
                          << " conf=" << b.confidence
                          << " box=(" << b.left << "," << b.top
                          << "," << b.right << "," << b.bottom << ")\n";
            }

            const auto img = r[0].image();
            check(img.data != nullptr,        "[1] image view has data");
            check(img.width > 0 && img.height > 0, "[1] image view has dims");
            check(img.channels == 3,          "[1] image view channels == 3");
            std::cout << "       image: " << img.width << "x" << img.height << "\n";
        }

        // ---------------------------------------------------------------
        // [2] 覆盖 conf_thresh = 0.9
        // ---------------------------------------------------------------
        step("2. override conf_thresh = 0.9");
        {
            trt_alpha::InferParams p;
            p.model_type  = trt_alpha::ModelType::yolov8;
            p.config_path = "configs/yolov8.ini";
            p.source      = imagePath;
            p.conf_thresh = 0.9f;

            trt_alpha::Infer model(p);
            trt_alpha::Result r = model.run();

            const std::size_t n = r[0].boxes().size();
            std::cout << "       boxes at conf=0.9: " << n << "\n";
            check(!r.empty(), "[2] result non-empty");
        }

        // ---------------------------------------------------------------
        // [3] frames() vs boxes()
        // ---------------------------------------------------------------
        step("3. frames() vs boxes() consistency");
        {
            trt_alpha::InferParams p;
            p.model_type  = trt_alpha::ModelType::yolov8;
            p.config_path = "configs/yolov8.ini";
            p.source      = imagePath;

            trt_alpha::Infer model(p);
            trt_alpha::Result r = model.run();

            const auto frames = r.frames();
            std::size_t viaFrames = 0;
            for (const auto& f : frames) { viaFrames += f.boxes().size(); }
            const std::size_t viaFlat = r.boxes().size();

            std::cout << "       via frames: " << viaFrames
                      << ", via flat: " << viaFlat << "\n";
            check(viaFrames == viaFlat, "[3] frames().boxes() == boxes()");
        }

        // ---------------------------------------------------------------
        // [4] 异步 async()：视频
        // ---------------------------------------------------------------
        step("4. async() on video");
        {
            trt_alpha::InferParams p;
            p.model_type  = trt_alpha::ModelType::yolov8;
            p.config_path = "configs/yolov8.ini";
            p.source      = videoPath;

            trt_alpha::Infer model(p);
            auto stream = model.async();

            int batchCount = 0;
            int frameCount = 0;
            trt_alpha::Result r;
            while (stream.get(r))
            {
                ++batchCount;
                frameCount += r.valid_count();
                if (batchCount >= 10) { break; }
            }
            stream.stop();

            std::cout << "       async batches: " << batchCount
                      << ", frames: " << frameCount << "\n";
            check(batchCount > 0, "[4] async got at least 1 batch");
            check(frameCount > 0, "[4] async got at least 1 frame");
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "[FAIL] exception: " << e.what() << "\n";
        ++g_failures;
    }

    std::cout << "===================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}