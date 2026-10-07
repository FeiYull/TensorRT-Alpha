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
#include <cctype>
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

//! 路径比较键：规范化 + 统一分隔符（+ Windows 下忽略大小写）。
//! 为什么要它：输出路径由 outputDir 拼出来，输入路径来自用户给的字符串，
//! 直接比字符串会把 "data\\bus.jpg" 和 "data/bus.jpg" 当成两个文件。
std::string pathKey(const fs::path& p)
{
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(p, ec);
    if (ec) { canon = p; }
    std::string s = canon.generic_string();
#ifdef _WIN32
    for (char& c : s) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
#endif
    return s;
}

//! 同上，但不做文件系统解析（用于"已是规范目录 + 单层文件名"的拼接结果）。
std::string lowerGeneric(const fs::path& p)
{
    std::string s = p.generic_string();
#ifdef _WIN32
    for (char& c : s) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
#endif
    return s;
}

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
    // 画关键点（如果有）——人脸 5 点等（实心点，白点半径 1）
    for (const auto& pt : det.land_marks)
    {
        cv::circle(image,
                   cv::Point(static_cast<int>(std::lround(pt.x)),
                             static_cast<int>(std::lround(pt.y))),
                   1, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA, 0);
    }
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

//! 画分割列表（先铺 mask，再画有框的框）。
void drawSegmentations(cv::Mat& image, const std::vector<seg::Segmentation>& segs,
                       const std::vector<core::ClassInfo>& classNames)
{
    for (const auto& s : segs)
    {
        blendMask(image, s);
    }
    for (const auto& s : segs)
    {
        if (s.box.label >= 0)   // label == -1 表示无框
        {
            drawBox(image, s.box, classNames);
        }
    }
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

// ---- COCO 17 关键点骨架 ----
//! 骨架：19 条边，顶点编号 1-based（COCO 标准）。
const std::vector<std::pair<int,int>>& cocoSkeleton()
{
    static const std::vector<std::pair<int,int>> kSkeleton = {
        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12},
        {7, 13},  {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},
        {2, 3},   {1, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6},
        {5, 7}
    };
    return kSkeleton;
}

//! 关键点颜色（17 个，BGR）。
const std::vector<cv::Scalar>& cocoKptColor()
{
    static const std::vector<cv::Scalar> kColor = {
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
        {0, 255, 0},    {0, 255, 0},    {255, 128, 0},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
        {255, 128, 0},  {255, 128, 0},  {51, 153, 255},
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
        {51, 153, 255}, {51, 153, 255}
    };
    return kColor;
}

//! 骨架颜色（19 条边，BGR）。
const std::vector<cv::Scalar>& cocoLimbColor()
{
    static const std::vector<cv::Scalar> kColor = {
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
        {255, 51, 255}, {255, 51, 255}, {255, 51, 255},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
        {255, 128, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0}
    };
    return kColor;
}

//! 画姿态（框 + 17 关键点 + 骨架）。
void drawKeypoints(cv::Mat& image, const std::vector<kpt::KeypointResult>& results)
{
    constexpr float kKptConfThresh = 0.5f;
    const auto& skeleton = cocoSkeleton();
    const auto& kptColors = cocoKptColor();
    const auto& limbColors = cocoLimbColor();

    for (const auto& kr : results)
    {
        // 画框（如果有 label >= 0）
        if (kr.box.label >= 0)
        {
            const int x0 = clampCoord(kr.box.left,   image.cols);
            const int y0 = clampCoord(kr.box.top,    image.rows);
            const int x1 = clampCoord(kr.box.right,  image.cols);
            const int y1 = clampCoord(kr.box.bottom, image.rows);
            if (x1 > x0 && y1 > y0)
            {
                cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1),
                              cv::Scalar(0, 255, 0), kBoxThickness, cv::LINE_AA);
            }
        }

        // 画关键点
        const int n = static_cast<int>(kr.keypoints.size());
        for (int k = 0; k < n; ++k)
        {
            const auto& kp = kr.keypoints[k];
            if (kp.confidence < kKptConfThresh) { continue; }
            const int kx = static_cast<int>(std::lround(kp.x));
            const int ky = static_cast<int>(std::lround(kp.y));
            if (kx < 0 || kx >= image.cols || ky < 0 || ky >= image.rows) { continue; }
            cv::circle(image, cv::Point(kx, ky), 5,
                       kptColors[k % kptColors.size()], cv::FILLED, cv::LINE_AA);
        }

        // 画骨架
        for (std::size_t si = 0; si < skeleton.size(); ++si)
        {
            const int a = skeleton[si].first - 1;   // 1-based -> 0-based
            const int b = skeleton[si].second - 1;
            if (a < 0 || a >= n || b < 0 || b >= n) { continue; }
            const auto& ka = kr.keypoints[a];
            const auto& kb = kr.keypoints[b];
            if (ka.confidence < kKptConfThresh || kb.confidence < kKptConfThresh) { continue; }
            cv::line(image,
                     cv::Point(static_cast<int>(std::lround(ka.x)),
                               static_cast<int>(std::lround(ka.y))),
                     cv::Point(static_cast<int>(std::lround(kb.x)),
                               static_cast<int>(std::lround(kb.y))),
                     limbColors[si % limbColors.size()], 2, cv::LINE_AA);
        }
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
        if (i < result.keypoints.size())
        {
            drawKeypoints(image, result.keypoints[i]);
        }
    }
}

void OpenCVRenderer::save(const core::BatchResult& result,
                          const std::string& outputDir) const
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

    // 输出目录规范化只做一次（每帧做一次 weakly_canonical 太贵）。
    // 之后每帧只需把文件名拼上去 —— 目录已规范，拼接结果无需再次解析。
    fs::path canonDir = fs::weakly_canonical(outputDir, ec);
    if (ec) { canonDir = fs::path(outputDir); }

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

        // 文件名：优先用数据源给的"帧来源名"（图片=原文件名）；
        // 缺失时回退 frame_<绝对帧号>，保证永不写出空名。
        const std::uint64_t index = result.firstFrameIndex + i;
        const std::string stem =
            (i < result.frameNames.size() && !result.frameNames[i].empty())
                ? result.frameNames[i]
                : ("frame_" + std::to_string(index));

        const fs::path out = fs::path(outputDir) / (stem + ".jpg");

        // 同名目标 == 某个输入源文件时拒绝写盘：输出与输入同一路径意味着
        // 把用户的原图覆盖掉（不可逆）。守卫在 app 层也有一道（按源类型提前
        // 报错），这里是最靠后的闸门 —— 任何数据源、任何调用方式都躲不过。
        if (m_inputPaths.count(lowerGeneric(canonDir / (stem + ".jpg"))) != 0)
        {
            TRT_LOG_ERROR("OpenCVRenderer::save: refusing to overwrite an input file: "
                          << out.string() << " (pass another dir to --save)");
            continue;
        }

        // 同名已存在（不是本次输入）时允许覆盖（用户口径），但必须留痕
        std::error_code existsEc;
        if (fs::exists(out, existsEc))
        {
            TRT_LOG_WARN("OpenCVRenderer::save: overwriting existing file: "
                         << out.string());
        }

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

void OpenCVRenderer::addInputSource(const std::string& resolvedPath)
{
    if (resolvedPath.empty())
    {
        return;
    }
    std::error_code ec;
    const fs::path p(resolvedPath);
    if (fs::is_directory(p, ec))
    {
        for (const auto& entry : fs::directory_iterator(p, ec))
        {
            if (entry.is_regular_file())
            {
                m_inputPaths.insert(pathKey(entry.path()));
            }
        }
    }
    else if (fs::is_regular_file(p, ec))
    {
        m_inputPaths.insert(pathKey(p));
    }
    // 视频 / 相机 / 流：渲染器不产出"原文件名"，无需登记
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