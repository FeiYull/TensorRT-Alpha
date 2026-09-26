// =============================================================================
//  test/test_yolov8/test_yolov8.cpp
// -----------------------------------------------------------------------------
//  YoloV8 测试：
//    [1] 注册中心能创建
//    [2] init 失败（engine 不存在）抛异常
//    [3] 真推理（需要 engine + 图片）：
//          读图 -> 构造 Batch -> setBatch/preprocess/infer/postprocess ->
//          检查框数 -> 用 OpenCV 画框 -> 存盘
//
//  用法：
//    test_yolov8                                 # 只跑 [1][2]
//    test_yolov8 <engine.trt> <image>            # 跑 [1][2][3]
// =============================================================================
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/det/detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using trt_alpha::IModel;
using trt_alpha::ModelRegistry;
using trt_alpha::core::Batch;
using trt_alpha::core::BatchResult;
using trt_alpha::core::Buffer;
using trt_alpha::core::BufferView;
using trt_alpha::core::DataType;
using trt_alpha::core::MemorySpace;
using trt_alpha::core::ModelConfig;
using trt_alpha::det::Detection;
using trt_alpha::det::IDetector;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

//! 从 cv::Mat 构造 Batch（batch_size == 1）。
Batch batchFromMat(const cv::Mat& img, int batchSize)
{
    if (img.empty() || img.type() != CV_8UC3)
    {
        throw std::runtime_error("batchFromMat: expect non-empty CV_8UC3 image");
    }
    if (batchSize <= 0)
    {
        throw std::runtime_error("batchFromMat: batchSize must be > 0");
    }

    const int W = img.cols;
    const int H = img.rows;
    const int C = 3;
    const std::size_t oneFrame = static_cast<std::size_t>(W) * H * C;

    auto buffer = Buffer::createHost(W * batchSize, H, C, DataType::UInt8);

    std::memcpy(buffer->mutableData(), img.data, oneFrame);
    if (batchSize > 1)
    {
        std::memset(buffer->mutableData() + oneFrame, 0,
                    oneFrame * static_cast<std::size_t>(batchSize - 1));
    }

    Batch b;
    b.sourceId = 0;
    b.firstFrameIndex = 0;
    b.buffer = buffer;
    b.validCount = 1;

    for (int i = 0; i < batchSize; ++i)
    {
        BufferView v;
        v.data = buffer->data() + static_cast<std::size_t>(i) * oneFrame;
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

//! 固定调色板：按 label 循环取色（同一 label 恒定同色）。
cv::Scalar colorForLabel(int label)
{
    static const cv::Scalar kPalette[] = {
        {232, 162, 12},  {70, 195, 152},  {207, 92, 231},  {91, 168, 250},
        {67, 106, 233},  {23, 23, 226},   {5, 159, 18},    {203, 18, 140},
        {201, 83, 59},   {87, 13, 232},
    };
    constexpr int kCount = static_cast<int>(sizeof(kPalette) / sizeof(kPalette[0]));
    return kPalette[((label % kCount) + kCount) % kCount];
}

//! 把坐标裁进图像范围（防越界）。
int clampCoord(float v, int limit)
{
    const int iv = static_cast<int>(std::lround(v));
    return std::max(0, std::min(iv, limit));
}

//! 画检测框 + 标签。
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections,
                    const std::vector<std::string>& classNames = {})
{
    constexpr int kThickness = 2;
    constexpr double kFontScale = 0.5;
    constexpr int kFontFace = cv::FONT_HERSHEY_DUPLEX;
    constexpr int kFontThickness = 1;

    for (const auto& d : detections)
    {
        const int x0 = clampCoord(d.left,   image.cols);
        const int y0 = clampCoord(d.top,    image.rows);
        const int x1 = clampCoord(d.right,  image.cols);
        const int y1 = clampCoord(d.bottom, image.rows);
        if (x1 <= x0 || y1 <= y0)
        {
            continue;   // 框退化
        }

        const cv::Scalar color = colorForLabel(d.label);
        cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1),
                      color, kThickness, cv::LINE_AA);

        // 标签文字（"class N 0.95"）
        std::string label = "class " + std::to_string(d.label);
        if (d.label >= 0 && static_cast<std::size_t>(d.label) < classNames.size())
        {
            label = classNames[static_cast<std::size_t>(d.label)];
        }
        char buf[128];
        std::snprintf(buf, sizeof(buf), " %s %.2f", label.c_str(), d.confidence);
        const std::string text(buf);

        int baseLine = 0;
        const cv::Size textSize =
            cv::getTextSize(text, kFontFace, kFontScale, kFontThickness, &baseLine);
        const int badgeW = std::min(textSize.width + 4, image.cols);
        const int badgeH = std::min(textSize.height + baseLine + 2, image.rows);
        const int badgeX = std::max(0, std::min(x0, image.cols - badgeW));
        const int badgeY = (y0 - badgeH >= 0) ? (y0 - badgeH) : y0;

        cv::rectangle(image,
                      cv::Rect(badgeX, badgeY, badgeW, badgeH),
                      color, cv::FILLED);
        cv::putText(image, text,
                    cv::Point(badgeX + 2, badgeY + textSize.height),
                    kFontFace, kFontScale, cv::Scalar(255, 255, 255),
                    kFontThickness, cv::LINE_AA);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    std::cout << "=== YoloV8 tests ===\n";

    // [1] 注册中心能创建
    {
        bool ok = false;
        try
        {
            auto m = ModelRegistry::instance().create("yolov8");
            ok = (m != nullptr) && (m->name() == "yolov8");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] create threw: " << e.what() << "\n";
        }
        check(ok, "[1] 'yolov8' registered and created");
    }

    // [2] init 失败（engine 不存在）
    {
        ModelConfig cfg;
        cfg.engine = "/definitely/not/exist/yolov8_12345.trt";
        cfg.batchSize = 1;
        cfg.dstH = 640;
        cfg.dstW = 640;

        bool threw = false;
        try
        {
            auto m = ModelRegistry::instance().create("yolov8");
            m->init(cfg);
        }
        catch (const std::runtime_error& e)
        {
            threw = true;
            std::cout << "       expected exception: " << e.what() << "\n";
        }
        check(threw, "[2] init with missing engine throws");
    }

    // [3] 真推理 
    if (argc >= 3)
    {
        std::cout << "\n--- real inference ---\n";
        const std::string enginePath = argv[1];
        const std::string imagePath = argv[2];

        try
        {
            const cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (img.empty())
            {
                std::cout << "[FAIL] cannot read image: " << imagePath << "\n";
                ++g_failures;
            }
            else
            {
                std::cout << "       image loaded: " << img.cols << "x" << img.rows << "\n";

                ModelConfig cfg;
                cfg.engine = enginePath;
                cfg.batchSize = 1;
                cfg.dstH = 640;
                cfg.dstW = 640;
                cfg.extras["num_class"] = "80";
                cfg.extras["conf_thresh"] = "0.25";

                auto m = ModelRegistry::instance().create("yolov8");
                m->init(cfg);

                auto* detector = dynamic_cast<IDetector*>(m.get());
                check(detector != nullptr, "[3] model is IDetector");

                if (detector != nullptr)
                {
                    Batch batch = batchFromMat(img, 1);

                    m->setBatch(batch);
                    m->preprocess();
                    m->infer();
                    m->postprocess();

                    BatchResult result;
                    m->commitResult(result);

                    check(!result.detections.empty(), "[3] detections has 1 image entry");
                    if (!result.detections.empty())
                    {
                        const std::size_t n = result.detections[0].size();
                        std::cout << "       detections: " << n << "\n";
                        for (std::size_t i = 0; i < n && i < 5; ++i)
                        {
                            const auto& d = result.detections[0][i];
                            std::cout << "         [" << i << "] "
                                      << "label=" << d.label
                                      << " conf=" << d.confidence
                                      << " box=(" << d.left << "," << d.top
                                      << "," << d.right << "," << d.bottom << ")\n";
                        }
                        check(n > 0, "[3] at least 1 detection (real inference works)");

                        // ---- 用 OpenCV 画框 + 存盘 ----
                        cv::Mat canvas = img.clone();
                        drawDetections(canvas, result.detections[0]);

                        const std::string outPath = "test_yolov8_result.jpg";
                        if (cv::imwrite(outPath, canvas))
                        {
                            std::cout << "       saved: " << outPath << "\n";
                            check(true, "[3] result image saved");
                        }
                        else
                        {
                            std::cout << "       [FAIL] cannot write " << outPath << "\n";
                            ++g_failures;
                        }
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [3] exception: " << e.what() << "\n";
            ++g_failures;
        }
    }
    else
    {
        std::cout << "\n(no engine+image args; real inference test skipped)\n";
        std::cout << "  usage: test_yolov8 <engine.trt> <image>\n";
    }

    std::cout << "====================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}