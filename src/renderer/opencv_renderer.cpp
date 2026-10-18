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
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace trt_alpha::renderer {
namespace {

// ---- 渲染风格（内部固定）----
constexpr int    kBoxThickness = 2;
constexpr int    kFontFace     = cv::FONT_HERSHEY_DUPLEX;
constexpr double kFontScale    = 0.5;
constexpr int    kFontThickness = 1;

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

//! 上游 Colors(i, bgr=True) 取色（定义在本文件后面，这里先前置声明）。
cv::Scalar ultraBgr(int index);

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

//! 画一枚"类别 + 置信度"徽标，锚点是框的左上角像素。
//! 从 drawBox 原样抽出：det 与 obb 共用同一套视觉，避免两处各写一份。
void drawLabelBadge(cv::Mat& image, int x0, int y0,
                    const std::string& text, const cv::Scalar& color)
{
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

//! 画一个框 + 标签（显式指定颜色）。
void drawBoxColored(cv::Mat& image, const det::Detection& det,
                    const std::vector<core::ClassInfo>& classNames,
                    const cv::Scalar& color)
{
    const int x0 = clampCoord(det.left,   image.cols);
    const int y0 = clampCoord(det.top,    image.rows);
    const int x1 = clampCoord(det.right,  image.cols);
    const int y1 = clampCoord(det.bottom, image.rows);
    if (x1 <= x0 || y1 <= y0)
    {
        return;   // 框退化
    }

    cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1),
                  color, kBoxThickness, cv::LINE_AA);

    // 标签
    drawLabelBadge(image, x0, y0,
                   labelName(det.label, classNames) + " " +
                       cv::format("%.2f", det.confidence),
                   color);
    // 画关键点（如果有）——人脸 5 点等（实心点，白点半径 1）
    for (const auto& pt : det.land_marks)
    {
        cv::circle(image,
                   cv::Point(static_cast<int>(std::lround(pt.x)),
                             static_cast<int>(std::lround(pt.y))),
                   1, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA, 0);
    }
}

//! 画一个框 + 标签（默认按 label 取色）。
void drawBox(cv::Mat& image, const det::Detection& det,
             const std::vector<core::ClassInfo>& classNames)
{
    drawBoxColored(image, det, classNames, colorForLabel(det.label));
}

//! 一维双线性取样表（ATen `align_corners=False` 口径）。
//!
//! 一帧里每个实例的 letterbox 几何和目标尺寸都一样，所以这组系数每帧只需算一次，
//! 而不是"每实例、每像素"各算一次。表里的每个值都是原逐像素写法里同名中间量的
//! 逐位复制，因此换成查表后结果不变。
struct AxisTaps
{
    std::vector<int>   i0;   //!< 第一个取样下标
    std::vector<int>   i1;   //!< 第二个取样下标（已夹到 netLen-1）
    std::vector<float> w0;   //!< i0 的权重
    std::vector<float> w1;   //!< i1 的权重
};

AxisTaps buildTaps(int srcLen, int netLen)
{
    AxisTaps t;
    const std::size_t n = static_cast<std::size_t>(srcLen);
    t.i0.resize(n);
    t.i1.resize(n);
    t.w0.resize(n);
    t.w1.resize(n);

    const float r = static_cast<float>(netLen) / static_cast<float>(srcLen);
    for (int i = 0; i < srcLen; ++i)
    {
        float s = r * (static_cast<float>(i) + 0.5f) - 0.5f;
        if (s < 0.0f) { s = 0.0f; }
        const int   a  = static_cast<int>(s);
        const int   b  = std::min(a + 1, netLen - 1);
        const float w1 = s - static_cast<float>(a);
        t.i0[i] = a;
        t.i1[i] = b;
        t.w1[i] = w1;
        t.w0[i] = 1.0f - w1;
    }
    return t;
}

//! 一张实例掩码放大到源图尺寸后的结果。
//!   img     : srcH x srcW 的 0/255 单通道
//!   rows    : 有前景的行号（升序）；其余行整行都是 0
//!   xlo/xhi : 该行首个/末个前景列（只在 rows 里的行上有效）
//!
//! 记这两个范围是为了让合成阶段能跳过整行、以及跳过行内的空白段 —— 被跳过的地方
//! 掩码恒为 0，而 m == 0 时合成公式的每一项都是恒等操作（见 drawSegmentations）。
struct ScaledMask
{
    cv::Mat          img;
    std::vector<int> rows;
    std::vector<int> xlo;
    std::vector<int> xhi;
};

//! 把网络坐标的掩码反 letterbox 回源图 —— 复刻 ultralytics `ops.scale_masks(
//! mask, im.shape[:2])`（mode="bilinear"）再 `> 0.5` 二值化。
//!
//! 掩码停在"引擎输入尺寸"（如 640x640，含灰边），坐标也是网络坐标；直接贴到源图
//! 必然错位。上游 Annotator.masks 正是靠这一步把带灰边的掩码还原到原图。
//!
//! 裁剪窗 == computeLetterBox(srcW, srcH, netW, netH) 的内容区（见
//! scale_mask_yolov26.hpp 头部注释，与 ops.scale_masks 的 crop 逐字同口径）。
//!
//! 两遍分离：原写法
//!     t0 = wx0*v00 + wx1*v01 ;  t1 = wx0*v10 + wx1*v11 ;  val = wy0*t0 + wy1*t1
//! 本来就是可分离的，所以水平那一步可以从"每个目标像素算一次"提到"每条 crop 行
//! 算一次"：中间量完全相同（同一个表达式、同样 float32 舍入），乘加次数降到 1/4，
//! 两层内循环也都变成连续访存。
ScaledMask scaleMaskToSource(const seg::Segmentation& s, int srcW, int srcH,
                             const AxisTaps& tx, const AxisTaps& ty,
                             std::vector<float>& hbuf,
                             int newW, int newH, int padL, int padT)
{
    ScaledMask out;
    const int netW = s.mask.width;
    const int netH = s.mask.height;
    if (netW <= 0 || netH <= 0 || srcW <= 0 || srcH <= 0 || s.mask.channels != 1)
    {
        return out;
    }
    if (newW <= 0 || newH <= 0 || padL < 0 || padT < 0 ||
        padL + newW > netW || padT + newH > netH)
    {
        return out;
    }
    if (static_cast<std::size_t>(newH) * static_cast<std::size_t>(srcW) > hbuf.size())
    {
        return out;
    }

    const std::uint8_t* base   = s.mask.data;
    const std::size_t   stride = static_cast<std::size_t>(s.mask.stride);

    // ---- 水平：每条 crop 行只算一次（中间量与单遍写法的 t0/t1 逐位相同） ----
    for (int r = 0; r < newH; ++r)
    {
        const std::uint8_t* src =
            base + static_cast<std::size_t>(padT + r) * stride + static_cast<std::size_t>(padL);
        float* dst = hbuf.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(srcW);
        for (int x = 0; x < srcW; ++x)
        {
            const float v0 = src[tx.i0[static_cast<std::size_t>(x)]] != 0 ? 1.0f : 0.0f;
            const float v1 = src[tx.i1[static_cast<std::size_t>(x)]] != 0 ? 1.0f : 0.0f;
            dst[x] = tx.w0[static_cast<std::size_t>(x)] * v0 +
                     tx.w1[static_cast<std::size_t>(x)] * v1;
        }
    }

    // ---- 垂直：合并两行 + 二值化，顺手记下前景范围 ----
    out.img.create(srcH, srcW, CV_8UC1);
    out.rows.reserve(128);
    out.xlo.assign(static_cast<std::size_t>(srcH), 0);
    out.xhi.assign(static_cast<std::size_t>(srcH), 0);

    for (int y = 0; y < srcH; ++y)
    {
        const std::size_t uy = static_cast<std::size_t>(y);
        const float* r0 = hbuf.data() + ty.i0[uy] * static_cast<std::size_t>(srcW);
        const float* r1 = hbuf.data() + ty.i1[uy] * static_cast<std::size_t>(srcW);
        const float  wy0 = ty.w0[uy];
        const float  wy1 = ty.w1[uy];

        std::uint8_t* o  = out.img.ptr<std::uint8_t>(y);
        int           lo = -1;
        int           hi = -1;
        for (int x = 0; x < srcW; ++x)
        {
            if (wy0 * r0[x] + wy1 * r1[x] > 0.5f)
            {
                o[x] = 255;
                if (lo < 0) { lo = x; }
                hi = x;
            }
            else
            {
                o[x] = 0;
            }
        }
        if (lo >= 0)
        {
            out.rows.push_back(y);
            out.xlo[uy] = lo;
            out.xhi[uy] = hi;
        }
    }
    return out;
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

//! 画分割列表 —— 复刻 ultralytics Annotator.masks 的整帧合成：
//!   mcs = amax_i(mask_i * color_i)，inv = prod_i(1 - mask_i * alpha)，
//!   out = (im/255 * inv + mcs) * 255（最后截断，即 torch 的 .byte()）。
void drawSegmentations(cv::Mat& image, const std::vector<seg::Segmentation>& segs,
                       const std::vector<core::ClassInfo>& classNames,
                       float alpha)
{
    if (segs.empty() || image.empty())
    {
        return;
    }

    const int srcW = image.cols;
    const int srcH = image.rows;

    // ---- 几何：一帧里所有实例共用同一套 letterbox 逆变换，只算一次 ----
    int netW = 0, netH = 0;
    int newW = 0, newH = 0, padL = 0, padT = 0;
    AxisTaps           tx, ty;
    std::vector<float> hbuf;

    const auto refreshGeometry = [&](int mw, int mh)
    {
        netW = mw;
        netH = mh;
        // 裁剪窗（round = half-to-even，与 computeLetterBox 一致）
        const double gain = std::min(static_cast<double>(netH) / srcH,
                                     static_cast<double>(netW) / srcW);
        newW = static_cast<int>(std::nearbyint(srcW * gain));
        newH = static_cast<int>(std::nearbyint(srcH * gain));
        const double dw = static_cast<double>(netW) - newW;
        const double dh = static_cast<double>(netH) - newH;
        padL = static_cast<int>(std::nearbyint(dw / 2.0 - 0.1));
        padT = static_cast<int>(std::nearbyint(dh / 2.0 - 0.1));
        tx = buildTaps(srcW, newW);
        ty = buildTaps(srcH, newH);
        hbuf.assign(static_cast<std::size_t>(newH) * static_cast<std::size_t>(srcW),
                    0.0f);
    };

    std::vector<ScaledMask> scaled;
    std::vector<cv::Vec3f>  colors;
    scaled.reserve(segs.size());
    colors.reserve(segs.size());

    for (const auto& s : segs)
    {
        if (s.mask.width <= 0 || s.mask.height <= 0 || s.mask.channels != 1)
        {
            continue;   // 无掩码的实例：原实现同样跳过
        }
        if (s.mask.width != netW || s.mask.height != netH)
        {
            refreshGeometry(s.mask.width, s.mask.height);
        }
        ScaledMask m = scaleMaskToSource(s, srcW, srcH, tx, ty, hbuf,
                                         newW, newH, padL, padT);
        if (m.img.empty())
        {
            continue;
        }
        const cv::Scalar bgr = ultraBgr(s.box.label);
        scaled.push_back(std::move(m));
        colors.push_back(cv::Vec3f(static_cast<float>(bgr[0]) / 255.0f * alpha,
                                   static_cast<float>(bgr[1]) / 255.0f * alpha,
                                   static_cast<float>(bgr[2]) / 255.0f * alpha));
    }

    if (!scaled.empty())
    {
        // 每行被哪些 layer 覆盖。合成只在"本行确实有前景"的区间里做；其它地方
        // m 恒为 0，而 m == 0 时 max(mcs, 0) 与 *= (1 - 0*alpha) 都是恒等，
        // 最后那次 (x/255)*255 往返对 0..255 也逐位是恒等（实测 256 个值全中），
        // 所以跳过与不跳过结果完全相同。
        std::vector<std::vector<int>> rowLayers(static_cast<std::size_t>(srcH));
        for (std::size_t li = 0; li < scaled.size(); ++li)
        {
            for (int y : scaled[li].rows)
            {
                rowLayers[static_cast<std::size_t>(y)].push_back(
                    static_cast<int>(li));
            }
        }

        std::vector<float> mcs0(static_cast<std::size_t>(srcW));
        std::vector<float> mcs1(static_cast<std::size_t>(srcW));
        std::vector<float> mcs2(static_cast<std::size_t>(srcW));
        std::vector<float> opa(static_cast<std::size_t>(srcW));

        // m == 1 时 (1.0f - m * alpha) 的精确值（1.0f * alpha 就是 alpha）
        const float keepOne = 1.0f - alpha;

        for (int y = 0; y < srcH; ++y)
        {
            const std::vector<int>& active = rowLayers[static_cast<std::size_t>(y)];
            if (active.empty())
            {
                continue;
            }

            const std::size_t uy = static_cast<std::size_t>(y);
            int xa = srcW;
            int xb = -1;
            for (int li : active)
            {
                const ScaledMask& m = scaled[static_cast<std::size_t>(li)];
                xa = std::min(xa, m.xlo[uy]);
                xb = std::max(xb, m.xhi[uy]);
            }
            if (xb < xa)
            {
                continue;
            }

            for (int x = xa; x <= xb; ++x)
            {
                const std::size_t ux = static_cast<std::size_t>(x);
                mcs0[ux] = 0.0f;
                mcs1[ux] = 0.0f;
                mcs2[ux] = 0.0f;
                opa[ux]  = 1.0f;
            }

            for (int li : active)
            {
                const ScaledMask& m = scaled[static_cast<std::size_t>(li)];
                const cv::Vec3f&  c = colors[static_cast<std::size_t>(li)];
                const std::uint8_t* mp = m.img.ptr<std::uint8_t>(y);
                const int lo = m.xlo[uy];
                const int hi = m.xhi[uy];
                for (int x = lo; x <= hi; ++x)
                {
                    if (mp[x] == 0)
                    {
                        continue;   // m == 0：三项都是恒等，跳过
                    }
                    const std::size_t ux = static_cast<std::size_t>(x);
                    mcs0[ux] = std::max(mcs0[ux], c[0]);
                    mcs1[ux] = std::max(mcs1[ux], c[1]);
                    mcs2[ux] = std::max(mcs2[ux], c[2]);
                    opa[ux] *= keepOne;
                }
            }

            cv::Vec3b* px = image.ptr<cv::Vec3b>(y);
            for (int x = xa; x <= xb; ++x)
            {
                const std::size_t ux  = static_cast<std::size_t>(x);
                const float       inv = opa[ux];
                cv::Vec3b&        dst = px[x];
                const float im0 = static_cast<float>(dst[0]) / 255.0f;
                const float im1 = static_cast<float>(dst[1]) / 255.0f;
                const float im2 = static_cast<float>(dst[2]) / 255.0f;
                dst[0] = static_cast<std::uint8_t>((im0 * inv + mcs0[ux]) * 255.0f);
                dst[1] = static_cast<std::uint8_t>((im1 * inv + mcs1[ux]) * 255.0f);
                dst[2] = static_cast<std::uint8_t>((im2 * inv + mcs2[ux]) * 255.0f);
            }
        }
    }

    for (const auto& s : segs)
    {
        if (s.box.label >= 0)   // label == -1 表示无框
        {
            drawBoxColored(image, s.box, classNames, ultraBgr(s.box.label));
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

// =============================================================================
//  depth / semantic / obb —— 三套"官方口径"的绘制
// -----------------------------------------------------------------------------
//  这三类结果此前只有接口没有画面。口径全部取自 ultralytics 的真实版本
//  （ultralytics/utils/plotting.py + ultralytics/utils/ops.py），不是自创：
//
//    depth  Annotator.depth_map(depth, alpha=0.6, cmap="jet", mode="disparity")
//           -> colorize_depth(...) -> cv2.applyColorMap，再 addWeighted 叠加。
//    sem    Annotator.semantic_mask(mask, alpha=0.5, ignore_index=255)
//           -> Colors 调色板按 class id 取色，整图 overlay 后 addWeighted。
//    obb    Results.plot -> ops.xywhr2xyxyxyxy -> box_label(multi_points=True)
//           -> cv2.polylines(闭合四边形)，标签锚在 box[0]。
//  放在一起是因为它们共享同一件事：把"模型给的稠密/旋转结果"变成 BGR 画面。
// =============================================================================

// ---- ultralytics Colors 调色板（20 色，RGB；convert 自上游的 hex 表）----
constexpr std::uint8_t kUltraPalette[20][3] = {
    {  4,  42, 255}, { 11, 219, 235}, {243, 243, 243}, {  0, 223, 183},
    { 17,  31, 104}, {255, 111, 221}, {255,  68,  79}, {204, 237,   0},
    {  0, 243,  68}, {189,   0, 255}, {  0, 180, 255}, {221,   0, 186},
    {  0, 255, 255}, { 38, 192,   0}, {  1, 255, 179}, {125,  36, 255},
    {123,   0, 104}, {255,  27, 108}, {252, 109,  47}, {162, 255,  11},
};

//! Colors(i, bgr=True)：先取 palette[i % 20]，再 RGB -> BGR。
cv::Scalar ultraBgr(int index)
{
    const int k = ((index % 20) + 20) % 20;
    return cv::Scalar(kUltraPalette[k][2], kUltraPalette[k][1], kUltraPalette[k][0]);
}

// ---- depth：Spectral_r 锚点 + LUT ----
//! 上游 _SPECTRAL_R_ANCHORS（RGB，far -> near），11 个。
constexpr std::uint8_t kSpectralAnchors[11][3] = {
    { 94,  79, 162}, { 51, 135, 188}, {102, 194, 165}, {170, 220, 164},
    {230, 245, 152}, {255, 254, 190}, {254, 224, 139}, {253, 173,  96},
    {244, 109,  67}, {212,  61,  79}, {158,   1,  66},
};

//! 上游 _spectral_lut()：np.linspace(0, 10, 256) 线性插值 -> 256x1x3 BGR LUT，
//! 交给 cv2.applyColorMap 当成用户色表（上游就是为了摆脱 matplotlib 才这么做的）。
cv::Mat spectralLut()
{
    cv::Mat lut(256, 1, CV_8UC3);
    for (int k = 0; k < 256; ++k)
    {
        const double xs = 10.0 * static_cast<double>(k) / 255.0;   // np.linspace(0,10,256)
        int i = static_cast<int>(xs);
        if (i > 9) { i = 9; }                                     // np.clip(..., 0, 9)
        const double f = xs - static_cast<double>(i);
        cv::Vec3b bgr;
        for (int ch = 0; ch < 3; ++ch)
        {
            const double v = static_cast<double>(kSpectralAnchors[i][ch]) * (1.0 - f) +
                             static_cast<double>(kSpectralAnchors[i + 1][ch]) * f;
            // 上游 rgb.round().astype(uint8)[:, ::-1]（RGB -> BGR）。
            // 用 nearbyint 对齐 numpy 的"四舍六入五成双"。
            const auto q = static_cast<std::uint8_t>(
                std::nearbyint(v));
            bgr[2 - ch] = q;
        }
        lut.at<cv::Vec3b>(k, 0) = bgr;
    }
    return lut;
}

enum class DepthCmap { Jet, Spectral };
enum class DepthMode { Metric, Disparity };

//! numpy.percentile(v, q)（默认 method="linear"）。
//!
//! Performance: runtime depth on a 1080p frame means ~2.1e6 samples, and this is
//! called twice per frame. A full std::sort is O(n log n) -- measured to be the
//! dominant cost that made the depth video path unusable. nth_element is O(n)
//! and yields the exact same values ("the k-th smallest" is what percentile
//! reads anyway), so the bit-exact contract with numpy is unchanged.
//!
//! Note: v is permuted (nth_element reorders); the caller only uses it to read
//! the two percentiles and never relies on its order afterwards.
float numpyPercentile(std::vector<float>& v, double q)
{
    if (v.empty()) { return 0.0f; }
    const double pos = (q / 100.0) * static_cast<double>(v.size() - 1);
    const double lo = std::floor(pos);
    const std::size_t i0 = static_cast<std::size_t>(lo);
    const std::size_t i1 = std::min<std::size_t>(i0 + 1, v.size() - 1);

    // Two partial selections instead of one full sort; x0 is read out before the
    // second selection reshuffles the array.
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i0), v.end());
    const float x0 = v[i0];
    float x1 = x0;
    if (i1 != i0)
    {
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i1), v.end());
        x1 = v[i1];
    }

    const double frac = pos - lo;
    return static_cast<float>(static_cast<double>(x0) * (1.0 - frac) +
                              static_cast<double>(x1) * frac);
}

//! 复刻上游 colorize_depth()：返回 BGR 伪彩图，无效像素（<= 0）为全黑。
cv::Mat colorizeDepth(const depth::DepthMap& dm, DepthCmap cmap, DepthMode mode)
{
    const int h = dm.height;
    const int w = dm.width;

    cv::Mat v(h, w, CV_32F);
    // A 1080p depth map is ~2.1e6 samples and this runs once per frame, so keep
    // the scratch buffer alive across calls instead of reallocating ~8 MB each time.
    static thread_local std::vector<float> pool;
    pool.clear();
    pool.reserve(static_cast<std::size_t>(h) * static_cast<std::size_t>(w));
    for (int y = 0; y < h; ++y)
    {
        float* row = v.ptr<float>(y);
        const float* src = dm.data.data() + static_cast<std::size_t>(y) * w;
        for (int x = 0; x < w; ++x)
        {
            const float d = src[x];
            const bool valid = d > 0.0f;
            const float val = (mode == DepthMode::Disparity)
                                  ? (valid ? 1.0f / d : 0.0f)
                                  : d;
            row[x] = val;
            if (valid) { pool.push_back(val); }
        }
    }

    float vmin = 0.0f;
    float vmax = 1.0f;
    if (!pool.empty())
    {
        if (mode == DepthMode::Disparity)
        {
            vmin = numpyPercentile(pool, 2.0);      // 2nd percentile
            vmax = numpyPercentile(pool, 98.0);     // 98th percentile
        }
        else
        {
            vmin = *std::min_element(pool.begin(), pool.end());
            vmax = *std::max_element(pool.begin(), pool.end());
        }
    }
    if (vmax <= vmin) { vmax = vmin + 1e-6f; }

    cv::Mat idx(h, w, CV_8U);
    for (int y = 0; y < h; ++y)
    {
        const float* vr = v.ptr<float>(y);
        std::uint8_t* ir = idx.ptr<std::uint8_t>(y);
        for (int x = 0; x < w; ++x)
        {
            float dn = (vr[x] - vmin) / (vmax - vmin);
            dn = std::max(0.0f, std::min(dn, 1.0f));
            // 上游 (dn * 255).astype(uint8) 是截断。
            ir[x] = static_cast<std::uint8_t>(dn * 255.0f);
        }
    }

    cv::Mat heat;
    if (cmap == DepthCmap::Spectral)
    {
        static const cv::Mat lut = spectralLut();
        cv::applyColorMap(idx, heat, lut);
    }
    else
    {
        cv::applyColorMap(idx, heat, cv::COLORMAP_JET);
    }

    // color[~valid] = 0（上游显式把无效像素涂黑）。
    for (int y = 0; y < h; ++y)
    {
        const float* src = dm.data.data() + static_cast<std::size_t>(y) * w;
        cv::Vec3b* hr = heat.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x)
        {
            if (!(src[x] > 0.0f)) { hr[x] = cv::Vec3b(0, 0, 0); }
        }
    }
    return heat;
}

//! 深度伪彩叠加（上游 Annotator.depth_map 的默认档：jet / disparity / alpha 0.6）。
void drawDepths(cv::Mat& image, const depth::DepthMap& dm)
{
    if (dm.empty() || image.empty())
    {
        return;
    }
    cv::Mat heat = colorizeDepth(dm, DepthCmap::Jet, DepthMode::Disparity);
    if (heat.size() != image.size())
    {
        cv::resize(heat, heat, image.size());   // 上游 cv2.resize 默认 INTER_LINEAR
    }
    cv::addWeighted(image, 1.0 - 0.6, heat, 0.6, 0.0, image);
}

//! 语义分割上色（上游 Annotator.semantic_mask：alpha 0.5，ignore_index 255 涂黑）。
void drawSemantics(cv::Mat& image, const sem::SemanticMask& mask)
{
    if (mask.empty() || image.empty())
    {
        return;
    }
    // 上游假定掩码与图同尺寸（整图 overlay）。不同尺寸时这里选择"不画"而不是
    // 缩放着色 —— resize 类别图必须用 nearest，且未定义的行为不该悄悄发生。
    if (mask.width != image.cols || mask.height != image.rows)
    {
        TRT_LOG_WARN("OpenCVRenderer: semantic mask " << mask.width << "x" << mask.height
                      << " does not match the image " << image.cols << "x" << image.rows
                      << ", skipping the overlay");
        return;
    }

    cv::Mat overlay(image.size(), CV_8UC3);
    for (int y = 0; y < mask.height; ++y)
    {
        const std::uint8_t* src = mask.data.data() + static_cast<std::size_t>(y) * mask.width;
        cv::Vec3b* dst = overlay.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.width; ++x)
        {
            const int id = src[x];
            if (id == 255)
            {
                dst[x] = cv::Vec3b(0, 0, 0);   // ignore_index
            }
            else
            {
                const int k = ((id % 20) + 20) % 20;
                dst[x] = cv::Vec3b(kUltraPalette[k][2], kUltraPalette[k][1],
                                   kUltraPalette[k][0]);
            }
        }
    }
    cv::addWeighted(image, 1.0 - 0.5, overlay, 0.5, 0.0, image);
}

//! 旋转框（上游 Results.plot 的 obb 分支：xywhr2xyxyxyxy -> polylines）。
void drawOrbs(cv::Mat& image, const std::vector<obb::ObbDetection>& obbs,
              const std::vector<core::ClassInfo>& classNames)
{
    for (const auto& o : obbs)
    {
        if (o.w <= 0.0f || o.h <= 0.0f) { continue; }

        // ops.xywhr2xyxyxyxy：
        //   vec1 = ( w/2 cosA,  w/2 sinA)
        //   vec2 = (-h/2 sinA,  h/2 cosA)
        //   pt1 = c + vec1 + vec2, pt2 = c + vec1 - vec2,
        //   pt3 = c - vec1 - vec2, pt4 = c - vec1 + vec2
        const double cosA = std::cos(static_cast<double>(o.angle));
        const double sinA = std::sin(static_cast<double>(o.angle));
        const double v1x = o.w * 0.5 * cosA;
        const double v1y = o.w * 0.5 * sinA;
        const double v2x = -o.h * 0.5 * sinA;
        const double v2y = o.h * 0.5 * cosA;

        const double cx = o.cx;
        const double cy = o.cy;
        const cv::Point2d corner[4] = {
            {cx + v1x + v2x, cy + v1y + v2y},
            {cx + v1x - v2x, cy + v1y - v2y},
            {cx - v1x - v2x, cy - v1y - v2y},
            {cx - v1x + v2x, cy - v1y + v2y},
        };

        // 上游 box_label: np.asarray(box, dtype=int) —— 截断取整（与 numpy 一致）。
        std::vector<cv::Point> pts(4);
        for (int k = 0; k < 4; ++k)
        {
            pts[static_cast<std::size_t>(k)] =
                cv::Point(static_cast<int>(corner[k].x),
                          static_cast<int>(corner[k].y));
        }

        const cv::Scalar color = colorForLabel(o.label);
        cv::polylines(image, std::vector<std::vector<cv::Point>>{pts}, true,
                      color, kBoxThickness, cv::LINE_AA);

        // 标签锚在 box[0]（pt1），与上游一致。
        drawLabelBadge(image, pts[0].x, pts[0].y,
                       labelName(o.label, classNames) + " " +
                           cv::format("%.2f", o.confidence),
                       color);
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

//! 画姿态（框 + 标签 + 17 关键点 + 骨架）。
void drawKeypoints(cv::Mat& image, const std::vector<kpt::KeypointResult>& results,
                   const std::vector<core::ClassInfo>& classNames,
                   float kptConfThresh)
{
    const auto& skeleton = cocoSkeleton();
    const auto& kptColors = cocoKptColor();
    const auto& limbColors = cocoLimbColor();

    for (const auto& kr : results)
    {
        const cv::Scalar color = ultraBgr(kr.box.label);
        // 画框 + 标签（类别 + 置信度）
        if (kr.box.label >= 0)
        {
            drawBoxColored(image, kr.box, classNames, color);
        }

        // 画关键点
        const int n = static_cast<int>(kr.keypoints.size());
        for (int k = 0; k < n; ++k)
        {
            const auto& kp = kr.keypoints[k];
            if (kp.confidence < kptConfThresh) { continue; }
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
            if (ka.confidence < kptConfThresh || kb.confidence < kptConfThresh) { continue; }
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
                                const std::vector<core::ClassInfo>& classNames,
                                const RenderOptions& options) const
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
            drawSegmentations(image, result.segmentations[i], classNames, options.maskAlpha);
        }
        if (i < result.obbs.size())
        {
            drawOrbs(image, result.obbs[i], classNames);
        }
        if (i < result.classifications.size())
        {
            drawClassifications(image, result.classifications[i], classNames);
        }
        if (i < result.semantics.size())
        {
            drawSemantics(image, result.semantics[i]);
        }
        if (i < result.depths.size())
        {
            drawDepths(image, result.depths[i]);
        }
        if (i < result.keypoints.size())
        {
            drawKeypoints(image, result.keypoints[i], classNames, options.kptConfThresh);
        }
    }
}

void OpenCVRenderer::save(const core::BatchResult& result,
                          const std::string& outputDir) const
{
    // 输出目录准备（create_directories + weakly_canonical）每个目录只做一次。
    // save() 是 const、每帧都被调用，而且 --save 并行时会被多个渲染线程同时调用，
    // 所以缓存用 mutable + 互斥保护。失败也只尝试一次、只报一次，免得逐帧刷屏。
    fs::path canonDir;
    {
        std::lock_guard<std::mutex> lock(m_outDirMutex);
        if (!m_outDirReady || m_outDirPath != outputDir)
        {
            m_outDirPath = outputDir;
            m_outDirReady = true;
            m_outDirOk = false;
            m_outDirCanon.clear();
            m_outDirWarnedOverwrite = false;

            if (outputDir.empty())
            {
                TRT_LOG_WARN("OpenCVRenderer::save: empty outputDir, skipping");
            }
            else
            {
                std::error_code ec;
                fs::create_directories(outputDir, ec);
                if (ec)
                {
                    TRT_LOG_ERROR("OpenCVRenderer::save: cannot create dir " << outputDir
                                  << " (" << ec.message() << ")");
                }
                else
                {
                    // 目录规范化只做一次；之后每帧只需把文件名拼上去。
                    const fs::path canon = fs::weakly_canonical(outputDir, ec);
                    m_outDirCanon = ec ? outputDir : canon.string();
                    m_outDirOk = true;
                    TRT_LOG_INFO("OpenCVRenderer: saving into " << m_outDirCanon);
                }
            }
        }
        if (!m_outDirOk)
        {
            return;
        }
        canonDir = fs::path(m_outDirCanon);
    }

    const std::size_t n =
        std::min<std::size_t>(result.views.size(),
                              static_cast<std::size_t>(std::max(0, result.validCount)));

    // JPEG quality is fixed for the whole call -> build the encoder params once
    // instead of per frame. 95 is what imwrite uses when no params are passed,
    // so the default path is byte-for-byte unchanged.
    const std::vector<int> encodeParams{cv::IMWRITE_JPEG_QUALITY, m_jpegQuality};

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

        // 同名已存在（不是本次输入）时允许覆盖（用户口径），但必须留痕。
        // 整个视频重跑时每一帧都会命中 —— 逐帧刷屏没有意义，每个输出目录只报一次。
        std::error_code existsEc;
        if (fs::exists(out, existsEc))
        {
            std::lock_guard<std::mutex> lock(m_outDirMutex);
            if (!m_outDirWarnedOverwrite)
            {
                m_outDirWarnedOverwrite = true;
                TRT_LOG_WARN("OpenCVRenderer::save: overwriting existing file(s), e.g. "
                             << out.string() << " (reported once per output dir)");
            }
        }

        if (!cv::imwrite(out.string(), image, encodeParams))
        {
            TRT_LOG_ERROR("OpenCVRenderer::save: imwrite failed: " << out.string());
        }
        else
        {
            TRT_LOG_DEBUG("OpenCVRenderer: saved " << out.string());
        }
    }
}

void OpenCVRenderer::setJpegQuality(int quality)
{
    // 1..100 is what libjpeg accepts; 0 would silently mean "libjpeg default",
    // so clamp instead of passing it through -- a silently different quality is
    // exactly the kind of surprise this framework refuses to ship.
    int q = quality;
    if (q < 1)
    {
        TRT_LOG_WARN("OpenCVRenderer: jpeg quality " << quality
                     << " out of range [1, 100], clamped to 1");
        q = 1;
    }
    else if (q > 100)
    {
        TRT_LOG_WARN("OpenCVRenderer: jpeg quality " << quality
                     << " out of range [1, 100], clamped to 100");
        q = 100;
    }
    m_jpegQuality = q;
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
