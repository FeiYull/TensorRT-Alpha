// =============================================================================
//  trt_alpha :: renderer :: opencv_renderer（实现）
// =============================================================================
#include "trt_alpha/renderer/opencv_renderer.hpp"

#include "trt_alpha/core/logger.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace trt_alpha::renderer {
namespace {

// ---- 渲染风格（内部固定）----
constexpr int    kBoxThickness = 2;
constexpr int    kFontFace     = cv::FONT_HERSHEY_DUPLEX;
constexpr double kFontScale    = 0.5;
constexpr int    kFontThickness = 1;
constexpr float  kMaskAlpha    = 0.35f;

//! label -> 颜色（固定调色板循环）。
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

//! label -> 类别名；越界时退化为 "class<label>"。
std::string labelName(int label, const std::vector<core::ClassInfo>& classNames)
{
    if (label >= 0 && static_cast<std::size_t>(label) < classNames.size())
    {
        return classNames[static_cast<std::size_t>(label)].name;
    }
    return "class" + std::to_string(label);
}

//! 浮点坐标裁进 [0, limit]。
int clampCoord(float value, int limit) noexcept
{
    const int iv = static_cast<int>(std::lround(value));
    return std::max(0, std::min(iv, limit));
}

//! 从 BufferView 构造"可写" cv::Mat（零拷贝）。
//! 注意：const_cast 是刻意的 —— 渲染就是"改图"。
cv::Mat matFromView(const core::BufferView& view)
{
    if (view.data == nullptr || view.width <= 0 || view.height <= 0)
    {
        return {};
    }
    if (view.channels != 3)
    {
        // 只支持 BGR 8-bit 三通道；其他类型跳过
        return {};
    }
    return cv::Mat(view.height, view.width, CV_8UC3,
                   const_cast<std::uint8_t*>(view.data),
                   static_cast<std::size_t>(view.stride));
}

//! 画一个框 + 标签。
void drawBox(cv::Mat& image, const det::Detection& det,
             const std::vector<core::ClassInfo>& classNames)
{
    const int x0 = clampCoord(det.left,   image.cols);
    const int y0 = clampCoord(det.top,    image.rows);
    const int x1 = clampCoord(det.right,  image.cols);
    const int y1 = clampCoord(det.bottom, image.rows);
    if (x1 <= x0 || y1 <= y0)
    {
        return;   // 框退化
    }

    const cv::Scalar color = colorForLabel(det.label);
    cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1),
                  color, kBoxThickness, cv::LINE_AA);

    // 标签
    const std::string text = labelName(det.label, classNames) + " " +
                             cv::format("%.2f", det.confidence);
    int baseLine = 0;
    const cv::Size textSize =
        cv::getTextSize(text, kFontFace, kFontScale, kFontThickness, &baseLine);
    const int badgeW = std::min(textSize.width + 4, image.cols);
    const int badgeH = std::min(textSize.height + baseLine + 2, image.rows);
    const int badgeX = std::max(0, std::min(x0, image.cols - badgeW));
    const int badgeY = (y0 - badgeH >= 0) ? (y0 - badgeH) : y0;

    cv::rectangle(image, cv::Rect(badgeX, badgeY, badgeW, badgeH),
                  color, cv::FILLED);
    cv::putText(image, text,
                cv::Point(badgeX + 2, badgeY + textSize.height),
                kFontFace, kFontScale, cv::Scalar(255, 255, 255),
                kFontThickness, cv::LINE_AA);
}

//! 掩码半透明叠加。
void blendMask(cv::Mat& image, const seg::Segmentation& instance)
{
    if (instance.mask.empty() || image.empty())
    {
        return;
    }
    if (instance.mask.channels != 1)
    {
        return;   // 契约：掩码为单通道
    }

    const int boxX = static_cast<int>(std::lround(instance.box.left));
    const int boxY = static_cast<int>(std::lround(instance.box.top));
    const int maskX0 = std::max(0, -boxX);
    const int maskY0 = std::max(0, -boxY);
    const int maskX1 = std::min(instance.mask.width,  image.cols - boxX);
    const int maskY1 = std::min(instance.mask.height, image.rows - boxY);
    if (maskX1 <= maskX0 || maskY1 <= maskY0)
    {
        return;   // 框完全在图像外
    }

    const cv::Rect maskRect(maskX0, maskY0, maskX1 - maskX0, maskY1 - maskY0);
    cv::Mat maskMat(instance.mask.height, instance.mask.width, CV_8UC1,
                    const_cast<std::uint8_t*>(instance.mask.data),
                    static_cast<std::size_t>(instance.mask.stride));
    const cv::Mat maskRoi = maskMat(maskRect);

    cv::Mat imageRoi = image(
        cv::Rect(boxX + maskX0, boxY + maskY0, maskRect.width, maskRect.height));

    const cv::Scalar color = colorForLabel(instance.box.label);
    const cv::Mat colorLayer(imageRoi.size(), CV_8UC3, color);
    cv::Mat blended;
    cv::addWeighted(imageRoi, 1.0 - kMaskAlpha, colorLayer, kMaskAlpha, 0.0, blended);
    blended.copyTo(imageRoi, maskRoi);
}

//! 画一个检测列表（框 + 标签）。
void drawDetections(cv::Mat& image, const std::vector<det::Detection>& detections,
                    const std::vector<core::ClassInfo>& classNames)
{
    for (const auto& det : detections)
    {
        drawBox(image, det, classNames);
    }
}

//! 画分割列表（先掩码，再框 + 标签）。
void drawSegmentations(cv::Mat& image, const std::vector<seg::Segmentation>& segs,
                       const std::vector<core::ClassInfo>& classNames)
{
    for (const auto& s : segs) { blendMask(image, s); }   // 先铺掩码
    for (const auto& s : segs) { drawBox(image, s.box, classNames); }  // 再压框
}

//! 画分类条（左上角逐行）。
void drawClassifications(cv::Mat& image, const std::vector<cls::ClassScore>& scores,
                         const std::vector<core::ClassInfo>& classNames)
{
    constexpr int kOriginX = 8;
    constexpr int kOriginY = 24;
    constexpr int kLineHeight = 20;
    int y = kOriginY;
    for (const auto& s : scores)
    {
        if (y > image.rows) break;
        const std::string text = labelName(s.label, classNames) + " " +
                                 cv::format("%.4f", s.score);
        cv::putText(image, text, cv::Point(kOriginX, y), kFontFace, kFontScale,
                    colorForLabel(s.label), kFontThickness, cv::LINE_AA);
        y += kLineHeight;
    }
}

}  // namespace

void OpenCVRenderer::drawResult(core::BatchResult& result,
                                const std::vector<core::ClassInfo>& classNames) const
{
    const std::size_t n =
        std::min<std::size_t>(result.views.size(),
                              static_cast<std::size_t>(std::max(0, result.validCount)));
    for (std::size_t i = 0; i < n; ++i)
    {
        cv::Mat image = matFromView(result.views[i]);
        if (image.empty())
        {
            continue;
        }

        if (i < result.detections.size())
        {
            drawDetections(image, result.detections[i], classNames);
        }
        if (i < result.segmentations.size())
        {
            drawSegmentations(image, result.segmentations[i], classNames);
        }
        if (i < result.classifications.size())
        {
            drawClassifications(image, result.classifications[i], classNames);
        }
    }
}

void OpenCVRenderer::save(const core::BatchResult& result,
                          const std::string& outputDir,
                          const std::string& prefix) const
{
    if (outputDir.empty())
    {
        TRT_LOG_WARN("OpenCVRenderer::save: empty outputDir, skipping");
        return;
    }
    std::error_code ec;
    fs::create_directories(outputDir, ec);
    if (ec)
    {
        TRT_LOG_ERROR("OpenCVRenderer::save: cannot create dir " << outputDir
                      << " (" << ec.message() << ")");
        return;
    }

    const std::size_t n =
        std::min<std::size_t>(result.views.size(),
                              static_cast<std::size_t>(std::max(0, result.validCount)));
    for (std::size_t i = 0; i < n; ++i)
    {
        const cv::Mat image = matFromView(result.views[i]);
        if (image.empty())
        {
            continue;
        }
        const std::uint64_t index = result.firstFrameIndex + i;
        const fs::path out = fs::path(outputDir) /
                             (prefix + std::to_string(index) + ".jpg");
        if (!cv::imwrite(out.string(), image))
        {
            TRT_LOG_ERROR("OpenCVRenderer::save: imwrite failed: " << out.string());
        }
        else
        {
            TRT_LOG_INFO("OpenCVRenderer: saved " << out.string());
        }
    }
}

void OpenCVRenderer::show(const core::BatchResult& result,
                          const std::string& windowName) const
{
    if (result.views.empty() || result.validCount <= 0)
    {
        return;
    }
    const cv::Mat image = matFromView(result.views[0]);
    if (image.empty())
    {
        return;
    }
    cv::imshow(windowName, image);
    cv::waitKey(1);   // 不阻塞
}

}  // namespace trt_alpha::renderer