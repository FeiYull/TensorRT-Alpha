// =============================================================================
//  gl_renderer -- the OpenGL IRenderer implementation (GPU display)
// -----------------------------------------------------------------------------
//  See gl_renderer.hpp. The drawing primitives mirror the OpenCV renderer
//  (same palettes / same label text / same box thickness / same keypoint
//  skeleton), but are emitted as GPU primitives instead of pixel writes.
// =============================================================================
#include "trt_alpha/renderer/gl_renderer.hpp"

#include "gl_context.hpp"
#include "trt_alpha/core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_alpha::renderer {
namespace {

constexpr int kBoxThickness = 2;

struct Rgb { std::uint8_t r, g, b; };

//! label -> colour, 10-colour cycle (RGB; the OpenCV renderer keeps BGR).
const Rgb& colorForLabel(int label)
{
    static const Rgb kPalette[] = {
        { 12, 162, 232}, {152, 195,  70}, {231,  92, 207}, {250, 168,  91},
        {233, 106,  67}, {226,  23,  23}, { 18, 159,   5}, {140,  18, 203},
        { 59,  83, 201}, {232,  13,  87},
    };
    constexpr int kCount = static_cast<int>(sizeof(kPalette) / sizeof(kPalette[0]));
    return kPalette[((label % kCount) + kCount) % kCount];
}

//! ultralytics Colors(i) -- 20 colours, RGB (already RGB upstream).
const Rgb& ultraRgb(int index)
{
    static const Rgb kUltra[] = {
        {  4,  42, 255}, { 11, 219, 235}, {243, 243, 243}, {  0, 223, 183},
        { 17,  31, 104}, {255, 111, 221}, {255,  68,  79}, {204, 237,   0},
        {  0, 243,  68}, {189,   0, 255}, {  0, 180, 255}, {221,   0, 186},
        {  0, 255, 255}, { 38, 192,   0}, {  1, 255, 179}, {125,  36, 255},
        {123,   0, 104}, {255,  27, 108}, {252, 109,  47}, {162, 255,  11},
    };
    constexpr int kCount = static_cast<int>(sizeof(kUltra) / sizeof(kUltra[0]));
    return kUltra[((index % kCount) + kCount) % kCount];
}

//! COCO 17 keypoint colours (RGB) -- same split as the OpenCV renderer.
const Rgb& kptColor(int k)
{
    static const Rgb kColor[17] = {
        {  0, 255,   0}, {  0, 255,   0}, {  0, 255,   0}, {  0, 255,   0},
        {  0, 255,   0}, {  0, 128, 255}, {  0, 128, 255}, {  0, 128, 255},
        {  0, 128, 255}, {  0, 128, 255}, {  0, 128, 255}, {255, 153,  51},
        {255, 153,  51}, {255, 153,  51}, {255, 153,  51}, {255, 153,  51},
        {255, 153,  51},
    };
    return kColor[((k % 17) + 17) % 17];
}

//! COCO skeleton limb colours (RGB).
const Rgb& limbColor(int s)
{
    static const Rgb kColor[19] = {
        {255, 153,  51}, {255, 153,  51}, {255, 153,  51}, {255, 153,  51},
        {255,  51, 255}, {255,  51, 255}, {255,  51, 255}, {  0, 128, 255},
        {  0, 128, 255}, {  0, 128, 255}, {  0, 128, 255}, {  0, 128, 255},
        {  0, 255,   0}, {  0, 255,   0}, {  0, 255,   0}, {  0, 255,   0},
        {  0, 255,   0}, {  0, 255,   0}, {  0, 255,   0},
    };
    return kColor[((s % 19) + 19) % 19];
}

//! COCO 17 keypoint skeleton (1-based vertex indices, 19 edges).
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

int clampCoord(float value, int limit) noexcept
{
    const int iv = static_cast<int>(std::lround(value));
    return std::max(0, std::min(iv, limit));
}

std::string labelName(int label, const std::vector<core::ClassInfo>& classNames)
{
    if (label >= 0 && static_cast<std::size_t>(label) < classNames.size())
    {
        return classNames[static_cast<std::size_t>(label)].name;
    }
    return "class" + std::to_string(label);
}

std::string formatConf(float c)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", c);
    return buf;
}

// ---- primitives (frame-pixel coordinates) -----------------------------------
enum class Kind { Rect, Line, Text };

struct Prim
{
    Kind kind = Kind::Rect;
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    std::uint8_t r = 255, g = 255, b = 255;
    std::string text;
};

void pushRect(std::vector<Prim>& out, float x0, float y0, float x1, float y1, const Rgb& c)
{
    out.push_back({Kind::Rect, x0, y0, x1, y1, c.r, c.g, c.b, ""});
}
void pushLine(std::vector<Prim>& out, float x0, float y0, float x1, float y1, const Rgb& c)
{
    out.push_back({Kind::Line, x0, y0, x1, y1, c.r, c.g, c.b, ""});
}
void pushText(std::vector<Prim>& out, float x0, float y0, const std::string& s, const Rgb& c)
{
    out.push_back({Kind::Text, x0, y0, 0, 0, c.r, c.g, c.b, s});
}

void drawLabelBadge(std::vector<Prim>& out, int x0, int y0, int fw,
                    const std::string& text, const Rgb& color)
{
    const int tw = static_cast<int>(text.size()) *
                   GLContext::kGlyphAdvance * GLContext::kGlyphCellPx;
    const int bh = GLContext::kGlyphRows * GLContext::kGlyphCellPx + 2;
    const int bw = std::min(tw + 4, fw);
    const int bx = std::max(0, std::min(x0, fw - bw));
    const int by = (y0 - bh >= 0) ? (y0 - bh) : y0;

    pushRect(out, static_cast<float>(bx), static_cast<float>(by),
             static_cast<float>(bx + bw), static_cast<float>(by + bh), color);
    pushText(out, static_cast<float>(bx + 2), static_cast<float>(by + 1),
             text, Rgb{255, 255, 255});
}

void drawBox(std::vector<Prim>& out, const det::Detection& det, int fw, int fh,
             const std::vector<core::ClassInfo>& classNames, const Rgb& color)
{
    const int x0 = clampCoord(det.left, fw);
    const int y0 = clampCoord(det.top, fh);
    const int x1 = clampCoord(det.right, fw);
    const int y1 = clampCoord(det.bottom, fh);
    if (x1 <= x0 || y1 <= y0)
    {
        return;
    }

    const float t = static_cast<float>(kBoxThickness);
    pushRect(out, x0, y0, x1, y0 + t, color);
    pushRect(out, x0, y1 - t, x1, y1, color);
    pushRect(out, x0, y0, x0 + t, y1, color);
    pushRect(out, x1 - t, y0, x1, y1, color);

    drawLabelBadge(out, x0, y0, fw,
                   labelName(det.label, classNames) + " " + formatConf(det.confidence),
                   color);

    for (const auto& pt : det.land_marks)
    {
        const float kx = std::lround(pt.x);
        const float ky = std::lround(pt.y);
        pushRect(out, kx - 1, ky - 1, kx + 1, ky + 1, Rgb{255, 255, 255});
    }
}

void drawOrbs(std::vector<Prim>& out, const std::vector<obb::ObbDetection>& obbs,
              int fw, const std::vector<core::ClassInfo>& classNames)
{
    for (const auto& o : obbs)
    {
        if (o.w <= 0.0f || o.h <= 0.0f) { continue; }

        const double cosA = std::cos(static_cast<double>(o.angle));
        const double sinA = std::sin(static_cast<double>(o.angle));
        const double v1x = o.w * 0.5 * cosA;
        const double v1y = o.w * 0.5 * sinA;
        const double v2x = -o.h * 0.5 * sinA;
        const double v2y = o.h * 0.5 * cosA;
        const double cx = o.cx, cy = o.cy;

        const float corner[4][2] = {
            {static_cast<float>(cx + v1x + v2x), static_cast<float>(cy + v1y + v2y)},
            {static_cast<float>(cx + v1x - v2x), static_cast<float>(cy + v1y - v2y)},
            {static_cast<float>(cx - v1x - v2x), static_cast<float>(cy - v1y - v2y)},
            {static_cast<float>(cx - v1x + v2x), static_cast<float>(cy - v1y + v2y)},
        };

        const Rgb color = colorForLabel(o.label);
        for (int k = 0; k < 4; ++k)
        {
            const int b = (k + 1) % 4;
            pushLine(out, corner[k][0], corner[k][1], corner[b][0], corner[b][1], color);
        }
        drawLabelBadge(out, static_cast<int>(corner[0][0]),
                       static_cast<int>(corner[0][1]), fw,
                       labelName(o.label, classNames) + " " + formatConf(o.confidence),
                       color);
    }
}

void drawKeypoints(std::vector<Prim>& out, const std::vector<kpt::KeypointResult>& results,
                   int fw, int fh, const std::vector<core::ClassInfo>& classNames,
                   float thresh)
{
    const auto& skeleton = cocoSkeleton();
    for (const auto& kr : results)
    {
        const Rgb color = ultraRgb(kr.box.label);
        if (kr.box.label >= 0)
        {
            drawBox(out, kr.box, fw, fh, classNames, color);
        }

        const int n = static_cast<int>(kr.keypoints.size());
        for (int k = 0; k < n; ++k)
        {
            const auto& kp = kr.keypoints[static_cast<std::size_t>(k)];
            if (kp.confidence < thresh) { continue; }
            const float kx = std::lround(kp.x);
            const float ky = std::lround(kp.y);
            if (kx < 0 || kx >= fw || ky < 0 || ky >= fh) { continue; }
            const Rgb c = kptColor(k);
            pushRect(out, kx - 2, ky - 2, kx + 2, ky + 2, c);
        }

        for (std::size_t si = 0; si < skeleton.size(); ++si)
        {
            const int a = skeleton[si].first - 1;
            const int b = skeleton[si].second - 1;
            if (a < 0 || a >= n || b < 0 || b >= n) { continue; }
            const auto& ka = kr.keypoints[static_cast<std::size_t>(a)];
            const auto& kb = kr.keypoints[static_cast<std::size_t>(b)];
            if (ka.confidence < thresh || kb.confidence < thresh) { continue; }
            pushLine(out, std::lround(ka.x), std::lround(ka.y),
                     std::lround(kb.x), std::lround(kb.y),
                     limbColor(static_cast<int>(si)));
        }
    }
}

void drawClassifications(std::vector<Prim>& out, const std::vector<cls::ClassScore>& scores,
                         int fh, const std::vector<core::ClassInfo>& classNames)
{
    constexpr int kOriginX = 8;
    constexpr int kOriginY = 24;
    constexpr int kLineHeight = 20;
    int y = kOriginY;
    for (const auto& s : scores)
    {
        if (y > fh) { break; }
        pushText(out, kOriginX, y,
                 labelName(s.label, classNames) + " " + formatConf(s.score),
                 colorForLabel(s.label));
        y += kLineHeight;
    }
}

}  // namespace

struct GLRenderer::Impl
{
    explicit Impl(int tiles) : ctx(tiles), scene(static_cast<std::size_t>(tiles)) {}

    GLContext ctx;
    std::vector<std::vector<Prim>> scene;   //!< per tile, frame-pixel coords
    bool ready = false;
    bool warnedUnsupported = false;
};

GLRenderer::GLRenderer(int numTiles)
    : m_impl(std::make_unique<Impl>(std::max(1, numTiles)))
{
}

GLRenderer::~GLRenderer() = default;

void GLRenderer::drawResult(core::BatchResult& result,
                            const std::vector<core::ClassInfo>& classNames,
                            const RenderOptions& options) const
{
    if (result.hostReady)
    {
        result.hostReady->wait();
    }

    const int tile = result.sourceId;
    if (tile < 0 || tile >= static_cast<int>(m_impl->scene.size()))
    {
        TRT_LOG_WARN("GLRenderer: result sourceId " << result.sourceId
                     << " outside the " << m_impl->scene.size() << " tiles, skipped");
        return;
    }
    if (result.views.empty())
    {
        return;
    }

    const int fw = result.views[0].width;
    const int fh = result.views[0].height;
    if (fw <= 0 || fh <= 0)
    {
        return;
    }

    std::vector<Prim>& out = m_impl->scene[static_cast<std::size_t>(tile)];
    out.clear();

    if (!result.detections.empty() && !result.detections[0].empty())
    {
        for (const auto& det : result.detections[0])
        {
            drawBox(out, det, fw, fh, classNames, colorForLabel(det.label));
        }
    }
    if (!result.obbs.empty() && !result.obbs[0].empty())
    {
        drawOrbs(out, result.obbs[0], fw, classNames);
    }
    if (!result.keypoints.empty() && !result.keypoints[0].empty())
    {
        drawKeypoints(out, result.keypoints[0], fw, fh, classNames,
                      options.kptConfThresh);
    }
    if (!result.classifications.empty() && !result.classifications[0].empty())
    {
        drawClassifications(out, result.classifications[0], fh, classNames);
    }

    if ((!result.segmentations.empty() || !result.depths.empty() ||
         !result.semantics.empty()) && !m_impl->warnedUnsupported)
    {
        m_impl->warnedUnsupported = true;
        TRT_LOG_WARN("GLRenderer: segmentation / depth / semantic overlays are not "
                     "drawn yet (v1 draws det / obb / pose / cls only)");
    }
}

void GLRenderer::save(const core::BatchResult& /*result*/,
                      const std::string& /*outputDir*/) const
{
    // The GL backend draws on the GPU at show time, so there is no annotated
    // host image to write. Disk output is a follow-up (nvEnc). The CLI rejects
    // `--render gl --save` up front, so this only trips on direct misuse.
    throw std::runtime_error(
        "GLRenderer::save: not implemented (the GL backend supports --show only; "
        "use --render opencv for --save)");
}

void GLRenderer::show(const core::BatchResult& result,
                      const std::string& windowName) const
{
    if (result.hostReady)
    {
        result.hostReady->wait();
    }
    if (result.views.empty() || result.validCount <= 0)
    {
        return;
    }

    const int tile = result.sourceId;
    if (tile < 0 || tile >= static_cast<int>(m_impl->scene.size()))
    {
        return;
    }

    if (!m_impl->ready)
    {
        const std::string title = windowName.empty() ? "trt_alpha" : windowName;
        if (!m_impl->ctx.init(title))
        {
            throw std::runtime_error("GLRenderer: failed to initialise the OpenGL context");
        }
        m_impl->ready = true;
    }

    const core::BufferView& v = result.views[0];
    if (v.empty() || v.channels != 3)
    {
        return;
    }
    if (v.space != core::MemorySpace::Host)
    {
        // v1 uploads host frames; a device -> texture path (CUDA-GL interop) is
        // a follow-up and needs IRenderer::acceptsDevicePixels.
        TRT_LOG_WARN("GLRenderer: device-resident frame on source " << tile
                     << " not displayed (v1 uploads host frames only)");
        return;
    }

    m_impl->ctx.uploadFrame(tile, v.data, v.width, v.height, v.stride);
    m_impl->ctx.beginFrame();

    for (const Prim& p : m_impl->scene[static_cast<std::size_t>(tile)])
    {
        switch (p.kind)
        {
        case Kind::Rect:
            m_impl->ctx.drawRect(tile, p.x0, p.y0, p.x1, p.y1, p.r, p.g, p.b);
            break;
        case Kind::Line:
            m_impl->ctx.drawLine(tile, p.x0, p.y0, p.x1, p.y1, p.r, p.g, p.b);
            break;
        case Kind::Text:
            m_impl->ctx.drawText(tile, p.x0, p.y0, p.text, p.r, p.g, p.b);
            break;
        }
    }

    m_impl->ctx.present();
}

bool GLRenderer::shouldClose() const noexcept
{
    return m_impl->ctx.shouldClose();
}

void GLRenderer::onSourceEnded(int sourceId)
{
    if (sourceId < 0 || sourceId >= static_cast<int>(m_impl->scene.size()))
    {
        return;
    }
    // Drop the last frame's primitives (they belong to a stream that is gone)
    // and switch that tile to the "NO SIGNAL" placeholder.
    m_impl->scene[static_cast<std::size_t>(sourceId)].clear();
    m_impl->ctx.markNoSignal(sourceId);
}

void GLRenderer::pump() const
{
    if (!m_impl->ready)
    {
        return;
    }
    m_impl->ctx.beginFrame();
    m_impl->ctx.present();
}

}  // namespace trt_alpha::renderer
