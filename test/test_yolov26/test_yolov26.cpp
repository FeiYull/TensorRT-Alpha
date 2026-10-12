// =============================================================================
//  test/test_yolov26/test_yolov26.cpp
// -----------------------------------------------------------------------------
//  YoloV26 测试：
//    [1] 注册中心能创建
//    [2] init 失败（engine 不存在）抛异常
//    [3] 真推理（需要 engine + 图片）：
//          读图 -> 构造 Batch -> setBatch/preprocess/infer/postprocess ->
//          校验框数 / label 范围 / conf 下界 / 坐标落在图内 ->
//          用 OpenCV 画框 -> 存盘
//
//  用法：
//    test_yolov26                                 # 只跑 [1][2]
//    test_yolov26 <engine.trt> <image> [batch]    # 跑 [1][2][3]，batch 默认 2
//
//  说明：YOLOV26 的测试引擎是静态 batch（导出 ONNX 时写死），所以 batch
//  参数必须与引擎一致，否则 init 会显式报错。
// =============================================================================
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/det/detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

//! 从 cv::Mat 构造 Batch（第 0 帧 = 原图，其余帧清零）。
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
    std::cout << "=== YoloV26 tests ===\n";

    // [1] 注册中心能创建
    {
        bool ok = false;
        try
        {
            auto m = ModelRegistry::instance().create("yolov26");
            ok = (m != nullptr) && (m->name() == "yolov26");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] create threw: " << e.what() << "\n";
        }
        check(ok, "[1] 'yolov26' registered and created");
    }

    // [2] init 失败（engine 不存在）
    {
        ModelConfig cfg;
        cfg.engine = "/definitely/not/exist/yolov26_12345.trt";
        cfg.batchSize = 1;
        cfg.dstH = 640;
        cfg.dstW = 640;

        bool threw = false;
        try
        {
            auto m = ModelRegistry::instance().create("yolov26");
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
        const std::string imagePath  = argv[2];
        const int batchSize = (argc >= 4) ? std::max(1, std::atoi(argv[3])) : 2;

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
                std::cout << "       image loaded: " << img.cols << "x" << img.rows
                          << ", batch = " << batchSize << "\n";

                ModelConfig cfg;
                cfg.engine = enginePath;
                cfg.batchSize = batchSize;
                cfg.dstH = 640;
                cfg.dstW = 640;
                cfg.extras["num_class"]  = "80";
                cfg.extras["extra"]      = "0";
                cfg.extras["conf_thresh"] = "0.25";
                cfg.extras["iou_thresh"]  = "0.70";
                cfg.extras["top_k"]       = "300";

                auto m = ModelRegistry::instance().create("yolov26");
                m->init(cfg);

                auto* detector = dynamic_cast<IDetector*>(m.get());
                check(detector != nullptr, "[3] model is IDetector");

                if (detector != nullptr)
                {
                    Batch batch = batchFromMat(img, batchSize);

                    m->setBatch(batch);
                    m->preprocess();
                    m->infer();
                    m->postprocess();

                    BatchResult result;
                    m->commitResult(result);

                    check(!result.detections.empty(), "[3] detections has image entries");

                    if (!result.detections.empty())
                    {
                        const std::vector<Detection>& dets = result.detections[0];
                        const std::size_t n = dets.size();
                        std::cout << "       detections: " << n << "\n";

                        // 后续要跟 demo 的输出逐值比对，所以打印到 6 位小数
                        for (std::size_t i = 0; i < n; ++i)
                        {
                            const auto& d = dets[i];
                            std::printf("         [%zu] label=%d conf=%.6f box=%.6f %.6f %.6f %.6f\n",
                                        i, d.label, (double)d.confidence,
                                        (double)d.left, (double)d.top,
                                        (double)d.right, (double)d.bottom);
                        }

                        check(n > 0, "[3] at least 1 detection (real inference works)");

                        // 语义断言：label 落在 [0,num_class)、conf 高于阈值、坐标在图内
                        bool labelOk = true, confOk = true, boxOk = true;
                        for (const auto& d : dets)
                        {
                            if (d.label < 0 || d.label >= 80)        { labelOk = false; }
                            if (!(d.confidence > 0.25f) || d.confidence > 1.0f) { confOk = false; }
                            if (d.left < 0.f || d.top < 0.f ||
                                d.right > static_cast<float>(img.cols) ||
                                d.bottom > static_cast<float>(img.rows) ||
                                d.right <= d.left || d.bottom <= d.top)  { boxOk = false; }
                        }
                        check(labelOk, "[3] every label within [0, num_class)");
                        check(confOk,  "[3] every confidence above conf_thresh and <= 1");
                        check(boxOk,   "[3] every box inside the source image");

                        // ---- 用 OpenCV 画框 + 存盘 ----
                        cv::Mat canvas = img.clone();
                        drawDetections(canvas, dets);

                        const std::string outPath = "test_yolov26_result.jpg";
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
        std::cout << "  usage: test_yolov26 <engine.trt> <image> [batch]\n";
    }

    std::cout << "====================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}
