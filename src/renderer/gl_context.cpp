// =============================================================================
//  gl_context -- GPU display context (implementation)
// -----------------------------------------------------------------------------
//  See gl_context.hpp for the contract. Everything OpenGL / GLFW lives here.
// =============================================================================
#define GLFW_INCLUDE_NONE
#include "gl_context.hpp"

#include "bgr_to_rgba.hpp"
#include "gl_loader.hpp"
#include "trt_alpha/core/logger.hpp"

#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_alpha::renderer {
namespace {

// ---- shaders ----------------------------------------------------------------

const char* kVideoVs = R"(#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() { vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }
)";

const char* kVideoFs = R"(#version 330 core
in vec2 vUV;
out vec4 o;
uniform sampler2D uTex;
void main() { vec3 c = texture(uTex, vUV).rgb; o = vec4(c.bgr, 1.0); }
)";

const char* kSolidVs = R"(#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec3 aColor;
out vec3 vColor;
void main() { vColor = aColor; gl_Position = vec4(aPos, 0.0, 1.0); }
)";

const char* kSolidFs = R"(#version 330 core
in vec3 vColor;
out vec4 o;
void main() { o = vec4(vColor, 1.0); }
)";

const char* kTextVs = R"(#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
layout(location=2) in vec3 aColor;
out vec2 vUV;
out vec3 vColor;
void main() { vUV = aUV; vColor = aColor; gl_Position = vec4(aPos, 0.0, 1.0); }
)";

const char* kTextFs = R"(#version 330 core
in vec2 vUV;
in vec3 vColor;
out vec4 o;
uniform sampler2D uFont;
void main() { float a = texture(uFont, vUV).r; o = vec4(vColor, a); }
)";

// ---- 5x7 bitmap font (public-domain, ASCII 32..126) -------------------------
// Each glyph is 5 columns; each byte is one column, bit 0 (LSB) = top row.
constexpr std::uint8_t kFont5x7[95][5] = {
    {0x00,0x00,0x00,0x00,0x00}, {0x00,0x00,0x5F,0x00,0x00}, {0x00,0x07,0x00,0x07,0x00},
    {0x14,0x7F,0x14,0x7F,0x14}, {0x24,0x2A,0x7F,0x2A,0x12}, {0x23,0x13,0x08,0x64,0x62},
    {0x36,0x49,0x55,0x22,0x50}, {0x00,0x05,0x03,0x00,0x00}, {0x00,0x1C,0x22,0x41,0x00},
    {0x00,0x41,0x22,0x1C,0x00}, {0x14,0x08,0x3E,0x08,0x14}, {0x08,0x08,0x3E,0x08,0x08},
    {0x00,0x50,0x30,0x00,0x00}, {0x08,0x08,0x08,0x08,0x08}, {0x00,0x60,0x60,0x00,0x00},
    {0x20,0x10,0x08,0x04,0x02}, {0x3E,0x51,0x49,0x45,0x3E}, {0x00,0x42,0x7F,0x40,0x00},
    {0x42,0x61,0x51,0x49,0x46}, {0x21,0x41,0x45,0x4B,0x31}, {0x18,0x14,0x12,0x7F,0x10},
    {0x27,0x45,0x45,0x45,0x39}, {0x3C,0x4A,0x49,0x49,0x30}, {0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36}, {0x06,0x49,0x49,0x29,0x1E}, {0x00,0x36,0x36,0x00,0x00},
    {0x00,0x56,0x36,0x00,0x00}, {0x08,0x14,0x22,0x41,0x00}, {0x14,0x14,0x14,0x14,0x14},
    {0x00,0x41,0x22,0x14,0x08}, {0x02,0x01,0x51,0x09,0x06}, {0x32,0x49,0x79,0x41,0x3E},
    {0x7E,0x11,0x11,0x11,0x7E}, {0x7F,0x49,0x49,0x49,0x36}, {0x3E,0x41,0x41,0x41,0x22},
    {0x7F,0x41,0x41,0x22,0x1C}, {0x7F,0x49,0x49,0x49,0x41}, {0x7F,0x09,0x09,0x09,0x01},
    {0x3E,0x41,0x49,0x49,0x7A}, {0x7F,0x08,0x08,0x08,0x7F}, {0x00,0x41,0x7F,0x41,0x00},
    {0x20,0x40,0x41,0x3F,0x01}, {0x7F,0x08,0x14,0x22,0x41}, {0x7F,0x40,0x40,0x40,0x40},
    {0x7F,0x02,0x0C,0x02,0x7F}, {0x7F,0x04,0x08,0x10,0x7F}, {0x3E,0x41,0x41,0x41,0x3E},
    {0x7F,0x09,0x09,0x09,0x06}, {0x3E,0x41,0x51,0x21,0x5E}, {0x7F,0x09,0x19,0x29,0x46},
    {0x46,0x49,0x49,0x49,0x31}, {0x01,0x01,0x7F,0x01,0x01}, {0x3F,0x40,0x40,0x40,0x3F},
    {0x1F,0x20,0x40,0x20,0x1F}, {0x3F,0x40,0x38,0x40,0x3F}, {0x63,0x14,0x08,0x14,0x63},
    {0x07,0x08,0x70,0x08,0x07}, {0x61,0x51,0x49,0x45,0x43}, {0x00,0x7F,0x41,0x41,0x00},
    {0x02,0x04,0x08,0x10,0x20}, {0x00,0x41,0x41,0x7F,0x00}, {0x04,0x02,0x01,0x02,0x04},
    {0x40,0x40,0x40,0x40,0x40}, {0x00,0x01,0x02,0x04,0x00}, {0x20,0x54,0x54,0x54,0x78},
    {0x7F,0x48,0x44,0x44,0x38}, {0x38,0x44,0x44,0x44,0x20}, {0x38,0x44,0x44,0x48,0x7F},
    {0x38,0x54,0x54,0x54,0x18}, {0x08,0x7E,0x09,0x01,0x02}, {0x0C,0x52,0x52,0x52,0x3E},
    {0x7F,0x08,0x04,0x04,0x78}, {0x00,0x44,0x7D,0x40,0x00}, {0x20,0x40,0x44,0x3D,0x00},
    {0x7F,0x10,0x28,0x44,0x00}, {0x00,0x41,0x7F,0x40,0x00}, {0x7C,0x04,0x18,0x04,0x78},
    {0x7C,0x08,0x04,0x04,0x78}, {0x38,0x44,0x44,0x44,0x38}, {0x7C,0x14,0x14,0x14,0x08},
    {0x08,0x14,0x14,0x18,0x7C}, {0x7C,0x08,0x04,0x04,0x08}, {0x48,0x54,0x54,0x54,0x20},
    {0x04,0x3F,0x44,0x40,0x20}, {0x3C,0x40,0x40,0x20,0x7C}, {0x1C,0x20,0x40,0x20,0x1C},
    {0x3C,0x40,0x30,0x40,0x3C}, {0x44,0x28,0x10,0x28,0x44}, {0x0C,0x50,0x50,0x50,0x3C},
    {0x44,0x64,0x54,0x4C,0x44}, {0x00,0x08,0x36,0x41,0x00}, {0x00,0x00,0x7F,0x00,0x00},
    {0x00,0x41,0x36,0x08,0x00}, {0x08,0x04,0x08,0x10,0x08},
};

constexpr int kFontCols   = 16;                       // glyphs per atlas row
constexpr int kFontRows   = 6;                        // atlas rows
constexpr int kAtlasW     = kFontCols * 5;
constexpr int kAtlasH     = kFontRows * 7;

//! Vertex layouts (CPU-side batch buffers, all in NDC / normalized 0..1 colour).
struct SolidVertex { float x, y, r, g, b; };
struct TextVertex  { float x, y, u, v, r, g, b; };

}  // namespace

struct GLContext::Impl
{
    GLFWwindow* window = nullptr;
    int numTiles = 1;

    struct Tile
    {
        GLuint tex = 0;
        int texW = 0, texH = 0;   //!< allocated texture size
        int frameW = 0, frameH = 0; //!< last uploaded frame size
        float offX = 0, offY = 0, scale = 1; //!< letterbox layout (framebuffer px)
        bool noSignal = false;    //!< source ended -> draw a "NO SIGNAL" placeholder

        // ---- CUDA-GL interop (device upload path only) ----
        GLuint pbo = 0;                          //!< GL_PIXEL_UNPACK_BUFFER staging
        int pboW = 0, pboH = 0;                  //!< PBO size (0 = not allocated)
        cudaGraphicsResource* cuRes = nullptr;   //!< registered PBO
        int cuW = 0, cuH = 0;                    //!< PBO size at registration (0 = none)
    };
    std::vector<Tile> tiles;

    // programs
    GLuint progVideo = 0, progSolid = 0, progText = 0;
    GLint  locVideoTex = -1, locFontTex = -1;

    // buffers
    GLuint vaoVideo = 0, vboVideo = 0;
    GLuint vaoSolid = 0, vboSolid = 0;
    GLuint vaoText  = 0, vboText  = 0;
    GLuint fontTex  = 0;

    // CPU batch buffers
    std::vector<SolidVertex> lineVerts;
    std::vector<SolidVertex> fillVerts;
    std::vector<TextVertex>  textVerts;

    int fbw = 0, fbh = 0;
    int cols = 1, rows = 1;

    //! Zoomed tile (-1 = grid). A click on a tile zooms it to fill the window;
    //! a second click (anywhere) returns to the grid. Owned here because the
    //! GLFW mouse callback fires inside GLContext, not the renderer.
    int focusedTile = -1;

    bool warnedNonTight = false;

    //! Handle a left click in WINDOW coordinates (GLFW: top-left origin, y down).
    //! Zooms the clicked tile, or drops back to the grid when already zoomed.
    void handleClick(double winX, double winY)
    {
        if (fbw <= 0 || fbh <= 0)
        {
            return;
        }

        // Window size may differ from the framebuffer (HiDPI), so rescale the
        // cursor into framebuffer space first.
        int winW = 0, winH = 0;
        glfwGetWindowSize(window, &winW, &winH);
        if (winW <= 0 || winH <= 0) { winW = fbw; winH = fbh; }
        const double fx = winX * static_cast<double>(fbw) / static_cast<double>(winW);
        // No Y flip: both the GLFW cursor and the layout's offY grow downward
        // from the top (toNdc maps sy=0 -> top, sy=fbh -> bottom). Flipping here
        // would hit the vertically-mirrored tile (click row 0 -> row rows-1).
        const double fy = winY * static_cast<double>(fbh) / static_cast<double>(winH);

        if (focusedTile >= 0)
        {
            focusedTile = -1;   // any click while zoomed returns to the grid
            return;
        }

        for (int i = 0; i < numTiles; ++i)
        {
            const Tile& t = tiles[static_cast<std::size_t>(i)];
            if (t.frameW <= 0 || t.frameH <= 0)
            {
                continue;
            }
            const double x0 = t.offX, y0 = t.offY;
            const double x1 = t.offX + static_cast<double>(t.frameW) * t.scale;
            const double y1 = t.offY + static_cast<double>(t.frameH) * t.scale;
            if (fx >= x0 && fx < x1 && fy >= y0 && fy < y1)
            {
                focusedTile = i;
                return;
            }
        }
    }

    // CUDA-GL interop stream (created lazily on the first device upload). One
    // non-blocking stream serves every tile: map -> kernel -> unmap are all
    // ordered on it, so the unmapped texture is GL-ready without an explicit sync.
    cudaStream_t interopStream = nullptr;
    bool interopStreamReady = false;

    // ---- GL helpers ----
    //! Ensure tile `t` has an RGBA8 texture sized w x h, allocating / reallocating
    //! as needed. Returns true if the storage was (re)allocated this call.
    bool ensureVideoTexture(Tile& t, int w, int h)
    {
        const bool realloc = (t.texW != w || t.texH != h);
        if (t.tex == 0)
        {
            glGenTextures(1, &t.tex);
            glBindTexture(GL_TEXTURE_2D, t.tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        else
        {
            glBindTexture(GL_TEXTURE_2D, t.tex);
        }
        if (realloc)
        {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            t.texW = w;
            t.texH = h;
        }
        return realloc;
    }
    GLuint compileShader(GLenum type, const char* src, bool* okOut)
    {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            char log[1024] = {0};
            glGetShaderInfoLog(s, sizeof(log) - 1, nullptr, log);
            TRT_LOG_ERROR("GLContext: shader compile failed: " << log);
            glDeleteShader(s);
            *okOut = false;
            return 0;
        }
        *okOut = true;
        return s;
    }

    GLuint makeProgram(const char* vsSrc, const char* fsSrc)
    {
        bool ok = false;
        GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc, &ok);
        if (!ok) { return 0; }
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc, &ok);
        if (!ok) { glDeleteShader(vs); return 0; }

        GLuint p = glCreateProgram();
        glAttachShader(p, vs);
        glAttachShader(p, fs);
        glLinkProgram(p);
        GLint linked = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &linked);
        glDeleteShader(vs);
        glDeleteShader(fs);
        if (!linked)
        {
            char log[1024] = {0};
            glGetProgramInfoLog(p, sizeof(log) - 1, nullptr, log);
            TRT_LOG_ERROR("GLContext: program link failed: " << log);
            glDeleteProgram(p);
            return 0;
        }
        return p;
    }

    void buildFontAtlas()
    {
        std::vector<std::uint8_t> atlas(static_cast<std::size_t>(kAtlasW) * kAtlasH, 0);
        for (int g = 0; g < 95; ++g)
        {
            const int gx = (g % kFontCols) * 5;
            const int gy = (g / kFontCols) * 7;
            for (int c = 0; c < 5; ++c)
            {
                const std::uint8_t col = kFont5x7[g][c];
                for (int r = 0; r < 7; ++r)
                {
                    if ((col >> r) & 0x01)
                    {
                        atlas[static_cast<std::size_t>(gy + r) * kAtlasW +
                              static_cast<std::size_t>(gx + c)] = 255;
                    }
                }
            }
        }

        glGenTextures(1, &fontTex);
        glBindTexture(GL_TEXTURE_2D, fontTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, kAtlasW, kAtlasH, 0,
                     GL_RED, GL_UNSIGNED_BYTE, atlas.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    //! Recompute the letterbox layout from the framebuffer + frame sizes.
    void ensureLayout(int w, int h)
    {
        fbw = w; fbh = h;

        // Single-tile zoom: only the focused tile gets a layout (letterboxed to
        // the full window). The rest are zeroed out so they neither rasterize
        // (present() skips them) nor turn primitives into a bogus-scale draw.
        if (focusedTile >= 0 && focusedTile < numTiles)
        {
            for (int i = 0; i < numTiles; ++i)
            {
                tiles[static_cast<std::size_t>(i)].offX = 0;
                tiles[static_cast<std::size_t>(i)].offY = 0;
                tiles[static_cast<std::size_t>(i)].scale = 0;
            }
            Tile& t = tiles[static_cast<std::size_t>(focusedTile)];
            int fw = t.frameW, fh = t.frameH;
            if (fw <= 0 || fh <= 0) { fw = 1; fh = 1; }

            const float margin = 8.0f;
            const float cw = static_cast<float>(w) - 2 * margin;
            const float ch = static_cast<float>(h) - 2 * margin;
            const float s = std::min(cw / static_cast<float>(fw),
                                     ch / static_cast<float>(fh));
            const float dw = static_cast<float>(fw) * s;
            const float dh = static_cast<float>(fh) * s;
            t.offX = (static_cast<float>(w) - dw) * 0.5f;
            t.offY = (static_cast<float>(h) - dh) * 0.5f;
            t.scale = s;
            return;
        }

        cols = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(numTiles)))));
        rows = std::max(1, (numTiles + cols - 1) / cols);

        const float cellW = static_cast<float>(w) / cols;
        const float cellH = static_cast<float>(h) / rows;
        const float margin = 8.0f;

        for (int i = 0; i < numTiles; ++i)
        {
            const int col = i % cols;
            const int row = i / cols;
            const float cx0 = col * cellW + margin;
            const float cy0 = row * cellH + margin;
            const float cw = cellW - 2 * margin;
            const float ch = cellH - 2 * margin;

            int fw = tiles[static_cast<std::size_t>(i)].frameW;
            int fh = tiles[static_cast<std::size_t>(i)].frameH;
            if (fw <= 0 || fh <= 0) { fw = 1; fh = 1; }

            const float s = std::min(cw / static_cast<float>(fw),
                                     ch / static_cast<float>(fh));
            const float dw = fw * s;
            const float dh = fh * s;
            tiles[static_cast<std::size_t>(i)].offX = cx0 + (cw - dw) * 0.5f;
            tiles[static_cast<std::size_t>(i)].offY = cy0 + (ch - dh) * 0.5f;
            tiles[static_cast<std::size_t>(i)].scale = s;
        }
    }

    void toNdc(int tile, float px, float py, float& nx, float& ny) const
    {
        const Tile& t = tiles[static_cast<std::size_t>(tile)];
        const float sx = t.offX + px * t.scale;
        const float sy = t.offY + py * t.scale;
        nx = sx / (fbw * 0.5f) - 1.0f;
        ny = 1.0f - sy / (fbh * 0.5f);
    }

    //! Draw a centred text string directly in framebuffer coordinates (the
    //! NO SIGNAL placeholder; the label badge uses the frame-space drawText).
    void drawTextFramebuffer(float cx, float cy, const std::string& text,
                             int cellPx, std::uint8_t r, std::uint8_t g,
                             std::uint8_t b)
    {
        const float cell = static_cast<float>(cellPx);
        const float adv  = cell * GLContext::kGlyphAdvance;
        const float totalW = static_cast<float>(text.size()) * adv;
        const float totalH = cell * GLContext::kGlyphRows;
        const float x0 = cx - totalW * 0.5f;
        const float y0 = cy - totalH * 0.5f;
        const float cr = r / 255.0f, cg = g / 255.0f, cb = b / 255.0f;

        for (std::size_t i = 0; i < text.size(); ++i)
        {
            unsigned char ch = static_cast<unsigned char>(text[i]);
            int gidx = static_cast<int>(ch) - 32;
            if (gidx < 0 || gidx >= 95) { gidx = '?' - 32; }

            const float gx = static_cast<float>(gidx % kFontCols) * 5.0f / kAtlasW;
            const float gy = static_cast<float>(gidx / kFontCols) * 7.0f / kAtlasH;
            const float gw = 5.0f / kAtlasW;
            const float gh = 7.0f / kAtlasH;

            const float px = x0 + static_cast<float>(i) * adv;
            const float py = y0;
            const float pw = cell * 5.0f;
            const float ph = cell * 7.0f;

            float nx0 = px / (fbw * 0.5f) - 1.0f, ny0 = 1.0f - py / (fbh * 0.5f);
            float nx1 = (px + pw) / (fbw * 0.5f) - 1.0f, ny1 = 1.0f - (py + ph) / (fbh * 0.5f);

            const float u0 = gx, v0 = gy, u1 = gx + gw, v1 = gy + gh;
            textVerts.push_back({nx0, ny0, u0, v0, cr, cg, cb});
            textVerts.push_back({nx1, ny0, u1, v0, cr, cg, cb});
            textVerts.push_back({nx1, ny1, u1, v1, cr, cg, cb});
            textVerts.push_back({nx0, ny0, u0, v0, cr, cg, cb});
            textVerts.push_back({nx1, ny1, u1, v1, cr, cg, cb});
            textVerts.push_back({nx0, ny1, u0, v1, cr, cg, cb});
        }
    }

    //! Fill a source-ended tile with a dark plate + a centred "NO SIGNAL" label.
    void drawNoSignalTile(int tile)
    {
        const Tile& t = tiles[static_cast<std::size_t>(tile)];
        const float fw = (t.frameW > 0) ? static_cast<float>(t.frameW) : 1.0f;
        const float fh = (t.frameH > 0) ? static_cast<float>(t.frameH) : 1.0f;
        const float x0 = t.offX, y0 = t.offY;
        const float x1 = t.offX + fw * t.scale, y1 = t.offY + fh * t.scale;

        const float nx0 = x0 / (fbw * 0.5f) - 1.0f, ny0 = 1.0f - y0 / (fbh * 0.5f);
        const float nx1 = x1 / (fbw * 0.5f) - 1.0f, ny1 = 1.0f - y1 / (fbh * 0.5f);
        const float br = 0.05f, bg = 0.05f, bb = 0.07f;
        fillVerts.push_back({nx0, ny0, br, bg, bb});
        fillVerts.push_back({nx1, ny0, br, bg, bb});
        fillVerts.push_back({nx1, ny1, br, bg, bb});
        fillVerts.push_back({nx0, ny0, br, bg, bb});
        fillVerts.push_back({nx1, ny1, br, bg, bb});
        fillVerts.push_back({nx0, ny1, br, bg, bb});

        drawTextFramebuffer((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, "NO SIGNAL",
                            GLContext::kGlyphCellPx * 3, 220, 60, 60);
    }

    // ---- CUDA-GL interop helpers ----
    //! Create the interop stream once. Fails loudly (and returns false) so the
    //! caller can fall back instead of silently skipping the frame.
    bool ensureInteropStream()
    {
        if (interopStreamReady)
        {
            return true;
        }
        const cudaError_t e =
            cudaStreamCreateWithFlags(&interopStream, cudaStreamNonBlocking);
        if (e != cudaSuccess)
        {
            TRT_LOG_ERROR("GLContext: cudaStreamCreate failed: " << cudaGetErrorString(e));
            return false;
        }
        interopStreamReady = true;
        return true;
    }

    //! Drop the CUDA side of a tile: unregister the mapped PBO. Called before a
    //! PBO reallocation and at teardown. A stream sync first guarantees no kernel
    //! is still writing the buffer.
    void releaseTileInterop(Tile& t)
    {
        if (t.cuRes == nullptr)
        {
            return;
        }
        if (interopStream != nullptr)
        {
            cudaStreamSynchronize(interopStream);
        }
        cudaGraphicsUnregisterResource(t.cuRes);
        t.cuRes = nullptr;
        t.cuW = 0;
        t.cuH = 0;
    }

    //! Ensure tile `t` has a PBO sized w x h (RGBA8 = w * h * 4 bytes), allocating
    //! or reallocating as needed.
    void ensurePbo(Tile& t, int w, int h)
    {
        if (t.pbo == 0)
        {
            glGenBuffers(1, &t.pbo);
        }
        if (t.pboW != w || t.pboH != h)
        {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, t.pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER,
                         static_cast<trt_gl::GLsizeiptrT>(w) * h * 4,
                         nullptr, GL_STREAM_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            t.pboW = w;
            t.pboH = h;
        }
    }
};

GLContext::GLContext(int numTiles)
    : m_impl(std::make_unique<Impl>())
{
    m_impl->numTiles = std::max(1, numTiles);
    m_impl->tiles.resize(static_cast<std::size_t>(m_impl->numTiles));
}

GLContext::~GLContext()
{
    // Release the CUDA-GL interop FIRST: the registered PBOs live under the
    // window's GL context, so they must be unregistered before that context goes
    // away. The stream is a pure CUDA resource and would leak if not freed here.
    for (auto& t : m_impl->tiles)
    {
        m_impl->releaseTileInterop(t);
    }
    if (m_impl->interopStream != nullptr)
    {
        cudaStreamDestroy(m_impl->interopStream);
        m_impl->interopStream = nullptr;
    }

    // The GL objects (textures / buffers / programs) all live under the window's
    // context, so destroying the window frees them. This destructor runs on the
    // thread that owns the GLRenderer (the main thread), while the context was
    // made current on the render thread -- calling glDelete* here with no current
    // context would be a no-op at best. Let the context teardown do the work.
    if (m_impl->window != nullptr)
    {
        glfwDestroyWindow(m_impl->window);
        m_impl->window = nullptr;
    }
}

bool GLContext::init(const std::string& title)
{
    glfwSetErrorCallback([](int code, const char* desc) {
        TRT_LOG_ERROR("GLFW error " << code << ": " << desc);
    });

    if (!glfwInit())
    {
        TRT_LOG_ERROR("GLContext: glfwInit failed");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_impl->window = glfwCreateWindow(1280, 720, title.c_str(), nullptr, nullptr);
    if (m_impl->window == nullptr)
    {
        TRT_LOG_ERROR("GLContext: no GL 3.3 core context available");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(m_impl->window);
    glfwSwapInterval(0);

    // Click-to-zoom: a left click zooms the tile under the cursor; a second
    // click returns to the grid. The callback forwards window coordinates to
    // Impl::handleClick (which flips Y and hit-tests the current layout).
    glfwSetWindowUserPointer(m_impl->window, this);
    glfwSetMouseButtonCallback(
        m_impl->window,
        [](GLFWwindow* w, int button, int action, int /*mods*/) {
            if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
            {
                auto* self = static_cast<GLContext*>(glfwGetWindowUserPointer(w));
                if (self == nullptr)
                {
                    return;
                }
                double mx = 0, my = 0;
                glfwGetCursorPos(w, &mx, &my);
                self->m_impl->handleClick(mx, my);
            }
        });

    const char* missing = nullptr;
    if (!trt_gl::load(&missing))
    {
        TRT_LOG_ERROR("GLContext: GL entry point missing: "
                      << (missing != nullptr ? missing : "?"));
        glfwDestroyWindow(m_impl->window);
        m_impl->window = nullptr;
        glfwTerminate();
        return false;
    }

    TRT_LOG_INFO("GLContext: " << reinterpret_cast<const char*>(glGetString(GL_RENDERER))
                 << ", " << reinterpret_cast<const char*>(glGetString(GL_VERSION)));

    m_impl->progVideo = m_impl->makeProgram(kVideoVs, kVideoFs);
    m_impl->progSolid = m_impl->makeProgram(kSolidVs, kSolidFs);
    m_impl->progText  = m_impl->makeProgram(kTextVs, kTextFs);
    if (m_impl->progVideo == 0 || m_impl->progSolid == 0 || m_impl->progText == 0)
    {
        return false;
    }
    m_impl->locVideoTex = glGetUniformLocation(m_impl->progVideo, "uTex");
    m_impl->locFontTex  = glGetUniformLocation(m_impl->progText, "uFont");

    // video quad: pos + uv
    glGenVertexArrays(1, &m_impl->vaoVideo);
    glBindVertexArray(m_impl->vaoVideo);
    glGenBuffers(1, &m_impl->vboVideo);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vboVideo);
    glBufferData(GL_ARRAY_BUFFER, 6 * 4 * sizeof(float), nullptr, GL_STREAM_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          reinterpret_cast<const void*>(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // solid: pos + color
    glGenVertexArrays(1, &m_impl->vaoSolid);
    glBindVertexArray(m_impl->vaoSolid);
    glGenBuffers(1, &m_impl->vboSolid);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vboSolid);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STREAM_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(SolidVertex), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SolidVertex),
                          reinterpret_cast<const void*>(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // text: pos + uv + color
    glGenVertexArrays(1, &m_impl->vaoText);
    glBindVertexArray(m_impl->vaoText);
    glGenBuffers(1, &m_impl->vboText);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vboText);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STREAM_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex),
                          reinterpret_cast<const void*>(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(TextVertex),
                          reinterpret_cast<const void*>(4 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    m_impl->buildFontAtlas();

    TRT_LOG_INFO("GLContext: ready (" << m_impl->numTiles << " tiles)");
    return true;
}

bool GLContext::shouldClose() const
{
    if (m_impl->window == nullptr)
    {
        return false;
    }
    return glfwWindowShouldClose(m_impl->window) != 0;
}

void GLContext::markNoSignal(int tile)
{
    if (tile < 0 || tile >= m_impl->numTiles)
    {
        return;
    }
    m_impl->tiles[static_cast<std::size_t>(tile)].noSignal = true;
}

int GLContext::focusedTile() const
{
    return m_impl->focusedTile;
}

bool GLContext::probeInterop()
{
    // Build a throwaway HIDDEN window + GL context on the calling thread, ask
    // CUDA whether it can reach it, then tear everything down. This runs on the
    // host-copy policy's thread BEFORE the render thread exists, so it must NOT
    // reuse (or leave behind) the renderer's context: WGL on Windows ties
    // CUDA-GL interop to the thread that created the context, and handing that
    // context to the render thread deadlocks on the first register/map.
    if (!glfwInit())
    {
        TRT_LOG_ERROR("GLContext: probe glfwInit failed");
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);   // no flash

    GLFWwindow* probe = glfwCreateWindow(1, 1, "trt_alpha-probe", nullptr, nullptr);
    if (probe == nullptr)
    {
        TRT_LOG_ERROR("GLContext: probe could not create a GL 3.3 context");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(probe);

    unsigned int glDevCount = 0;
    const bool ok = (cudaGLGetDevices(&glDevCount, nullptr, 0, cudaGLDeviceListAll) ==
                         cudaSuccess &&
                     glDevCount > 0);
    TRT_LOG_INFO("GLContext: CUDA-GL interop "
                 << (ok ? "available" : "UNAVAILABLE (host copy)"));

    glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(probe);
    glfwTerminate();
    return ok;
}

void GLContext::uploadFrame(int tile, const std::uint8_t* data, int w, int h, int stride)
{
    if (tile < 0 || tile >= m_impl->numTiles || data == nullptr || w <= 0 || h <= 0)
    {
        return;
    }
    Impl::Tile& t = m_impl->tiles[static_cast<std::size_t>(tile)];

    if (stride != w * 3 && !m_impl->warnedNonTight)
    {
        m_impl->warnedNonTight = true;
        TRT_LOG_WARN("GLContext: frame stride " << stride << " != tight " << (w * 3)
                     << "; the GL backend expects tightly-packed BGR8 and may skew");
    }

    m_impl->ensureVideoTexture(t, w, h);

    // The bytes are BGR8; GL is told they are RGB (3 bytes/pixel) and expands
    // alpha to 1.0 into the RGBA8 texture. The video fragment shader swaps .bgr
    // back to true RGB. The device path below stores the same (B, G, R, 255)
    // texels, so one shader serves both.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, data);
    t.frameW = w;
    t.frameH = h;
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLContext::uploadFrameDevice(int tile, const std::uint8_t* devData, int w, int h, int stride)
{
    if (tile < 0 || tile >= m_impl->numTiles || devData == nullptr || w <= 0 || h <= 0)
    {
        return;
    }
    Impl::Tile& t = m_impl->tiles[static_cast<std::size_t>(tile)];

    if (!m_impl->ensureInteropStream())
    {
        return;
    }

    // The PBO must have storage before it can be registered; if it is about to
    // be (re)allocated, the old registration must go first -- it refers to the
    // buffer object that glBufferData is about to replace.
    const bool realloc = (t.pboW != w || t.pboH != h);
    if (realloc && t.cuRes != nullptr)
    {
        m_impl->releaseTileInterop(t);
    }
    m_impl->ensurePbo(t, w, h);

    if (t.cuRes == nullptr)
    {
        const cudaError_t e = cudaGraphicsGLRegisterBuffer(
            &t.cuRes, t.pbo, cudaGraphicsRegisterFlagsWriteDiscard);
        if (e != cudaSuccess)
        {
            TRT_LOG_ERROR("GLContext: cudaGraphicsGLRegisterBuffer failed: "
                          << cudaGetErrorString(e));
            return;
        }
        t.cuW = w;
        t.cuH = h;
    }

    cudaError_t e = cudaGraphicsMapResources(1, &t.cuRes, m_impl->interopStream);
    if (e != cudaSuccess)
    {
        TRT_LOG_ERROR("GLContext: cudaGraphicsMapResources failed: "
                      << cudaGetErrorString(e));
        return;
    }

    // A buffer resource maps to a plain writable device pointer -- no surface
    // object, no read-only cudaArray. This is the interop path that has always
    // worked across drivers/architectures.
    void* pboPtr = nullptr;
    std::size_t pboSize = 0;
    e = cudaGraphicsResourceGetMappedPointer(&pboPtr, &pboSize, t.cuRes);
    if (e != cudaSuccess)
    {
        TRT_LOG_ERROR("GLContext: cudaGraphicsResourceGetMappedPointer failed: "
                      << cudaGetErrorString(e));
        cudaGraphicsUnmapResources(1, &t.cuRes, m_impl->interopStream);
        return;
    }

    e = bgr8ToRgba(m_impl->interopStream, devData, stride,
                   static_cast<std::uint8_t*>(pboPtr), w, h);
    if (e != cudaSuccess)
    {
        TRT_LOG_ERROR("GLContext: bgr8ToRgba failed: " << cudaGetErrorString(e));
    }

    // Same stream as the kernel: the unmap's bookkeeping is ordered after it, so
    // the PBO is GL-ready for the glTexSubImage2D below -- no explicit sync.
    cudaGraphicsUnmapResources(1, &t.cuRes, m_impl->interopStream);

    // DMA the PBO into the texture (still on the GPU). The texture is bound by
    // ensureVideoTexture below.
    m_impl->ensureVideoTexture(t, w, h);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, t.pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    t.frameW = w;
    t.frameH = h;
}

void GLContext::drawRect(int tile, float x0, float y0, float x1, float y1,
                         std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
    if (tile < 0 || tile >= m_impl->numTiles || x1 <= x0 || y1 <= y0)
    {
        return;
    }
    const float cr = r / 255.0f, cg = g / 255.0f, cb = b / 255.0f;
    float nx[4], ny[4];
    m_impl->toNdc(tile, x0, y0, nx[0], ny[0]);
    m_impl->toNdc(tile, x1, y0, nx[1], ny[1]);
    m_impl->toNdc(tile, x1, y1, nx[2], ny[2]);
    m_impl->toNdc(tile, x0, y1, nx[3], ny[3]);

    // two triangles: 0-1-2, 0-2-3
    m_impl->fillVerts.push_back({nx[0], ny[0], cr, cg, cb});
    m_impl->fillVerts.push_back({nx[1], ny[1], cr, cg, cb});
    m_impl->fillVerts.push_back({nx[2], ny[2], cr, cg, cb});
    m_impl->fillVerts.push_back({nx[0], ny[0], cr, cg, cb});
    m_impl->fillVerts.push_back({nx[2], ny[2], cr, cg, cb});
    m_impl->fillVerts.push_back({nx[3], ny[3], cr, cg, cb});
}

void GLContext::drawLine(int tile, float x0, float y0, float x1, float y1,
                         std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
    if (tile < 0 || tile >= m_impl->numTiles)
    {
        return;
    }
    const float cr = r / 255.0f, cg = g / 255.0f, cb = b / 255.0f;
    float nx0, ny0, nx1, ny1;
    m_impl->toNdc(tile, x0, y0, nx0, ny0);
    m_impl->toNdc(tile, x1, y1, nx1, ny1);
    m_impl->lineVerts.push_back({nx0, ny0, cr, cg, cb});
    m_impl->lineVerts.push_back({nx1, ny1, cr, cg, cb});
}

void GLContext::drawText(int tile, float x0, float y0, const std::string& text,
                         std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
    if (tile < 0 || tile >= m_impl->numTiles || text.empty())
    {
        return;
    }
    const float cr = r / 255.0f, cg = g / 255.0f, cb = b / 255.0f;
    const float cell = static_cast<float>(kGlyphCellPx);
    const float adv = cell * kGlyphAdvance;

    for (std::size_t i = 0; i < text.size(); ++i)
    {
        unsigned char ch = static_cast<unsigned char>(text[i]);
        int g = static_cast<int>(ch) - 32;
        if (g < 0 || g >= 95) { g = '?' - 32; }

        const float gx = static_cast<float>(g % kFontCols) * 5.0f / kAtlasW;
        const float gy = static_cast<float>(g / kFontCols) * 7.0f / kAtlasH;
        const float gw = 5.0f / kAtlasW;
        const float gh = 7.0f / kAtlasH;

        const float px = x0 + static_cast<float>(i) * adv;
        const float py = y0;
        const float pw = cell * 5.0f;
        const float ph = cell * 7.0f;

        float nx[4], ny[4];
        m_impl->toNdc(tile, px, py, nx[0], ny[0]);
        m_impl->toNdc(tile, px + pw, py, nx[1], ny[1]);
        m_impl->toNdc(tile, px + pw, py + ph, nx[2], ny[2]);
        m_impl->toNdc(tile, px, py + ph, nx[3], ny[3]);

        const float u0 = gx, v0 = gy, u1 = gx + gw, v1 = gy + gh;
        m_impl->textVerts.push_back({nx[0], ny[0], u0, v0, cr, cg, cb});
        m_impl->textVerts.push_back({nx[1], ny[1], u1, v0, cr, cg, cb});
        m_impl->textVerts.push_back({nx[2], ny[2], u1, v1, cr, cg, cb});
        m_impl->textVerts.push_back({nx[0], ny[0], u0, v0, cr, cg, cb});
        m_impl->textVerts.push_back({nx[2], ny[2], u1, v1, cr, cg, cb});
        m_impl->textVerts.push_back({nx[3], ny[3], u0, v1, cr, cg, cb});
    }
}

void GLContext::beginFrame()
{
    if (m_impl->window == nullptr)
    {
        return;
    }
    glfwPollEvents();
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_impl->window, &w, &h);
    if (w > 0 && h > 0)
    {
        m_impl->ensureLayout(w, h);
    }
    m_impl->lineVerts.clear();
    m_impl->fillVerts.clear();
    m_impl->textVerts.clear();
}

void GLContext::present()
{
    if (m_impl->window == nullptr)
    {
        return;
    }
    glfwPollEvents();
    if (shouldClose())
    {
        return;
    }
    if (glfwGetKey(m_impl->window, GLFW_KEY_ESCAPE) == GLFW_PRESS ||
        glfwGetKey(m_impl->window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(m_impl->window, GLFW_TRUE);
        return;
    }

    int w = 0, h = 0;
    glfwGetFramebufferSize(m_impl->window, &w, &h);
    if (w <= 0 || h <= 0) { return; }
    m_impl->ensureLayout(w, h);

    glViewport(0, 0, w, h);
    glClearColor(0.10f, 0.10f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // ---- video quads (one draw per tile: each has its own texture) ----
    glUseProgram(m_impl->progVideo);
    glUniform1i(m_impl->locVideoTex, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(m_impl->vaoVideo);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vboVideo);
    for (int i = 0; i < m_impl->numTiles; ++i)
    {
        // Zoomed: rasterize only the focused tile (its layout fills the window).
        if (m_impl->focusedTile >= 0 && i != m_impl->focusedTile)
        {
            continue;
        }
        Impl::Tile& t = m_impl->tiles[static_cast<std::size_t>(i)];
        if (t.noSignal)
        {
            m_impl->drawNoSignalTile(i);
            continue;
        }
        if (t.tex == 0 || t.frameW <= 0 || t.frameH <= 0)
        {
            continue;
        }
        const float fw = static_cast<float>(t.frameW);
        const float fh = static_cast<float>(t.frameH);
        float q[4][2];      // NDC
        float uv[4][2];
        m_impl->toNdc(i, 0, 0, q[0][0], q[0][1]);
        m_impl->toNdc(i, fw, 0, q[1][0], q[1][1]);
        m_impl->toNdc(i, fw, fh, q[2][0], q[2][1]);
        m_impl->toNdc(i, 0, fh, q[3][0], q[3][1]);
        uv[0][0] = 0; uv[0][1] = 0;
        uv[1][0] = 1; uv[1][1] = 0;
        uv[2][0] = 1; uv[2][1] = 1;
        uv[3][0] = 0; uv[3][1] = 1;

        // two triangles: 0-1-2, 0-2-3
        const float verts[6][4] = {
            {q[0][0], q[0][1], uv[0][0], uv[0][1]},
            {q[1][0], q[1][1], uv[1][0], uv[1][1]},
            {q[2][0], q[2][1], uv[2][0], uv[2][1]},
            {q[0][0], q[0][1], uv[0][0], uv[0][1]},
            {q[2][0], q[2][1], uv[2][0], uv[2][1]},
            {q[3][0], q[3][1], uv[3][0], uv[3][1]},
        };
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STREAM_DRAW);
        glBindTexture(GL_TEXTURE_2D, t.tex);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    // ---- solid: lines then fills ----
    glUseProgram(m_impl->progSolid);
    glBindVertexArray(m_impl->vaoSolid);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vboSolid);
    const std::size_t totalSolid = m_impl->lineVerts.size() + m_impl->fillVerts.size();
    if (totalSolid > 0)
    {
        std::vector<SolidVertex> all;
        all.reserve(totalSolid);
        all.insert(all.end(), m_impl->lineVerts.begin(), m_impl->lineVerts.end());
        all.insert(all.end(), m_impl->fillVerts.begin(), m_impl->fillVerts.end());
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<trt_gl::GLsizeiptrT>(all.size() * sizeof(SolidVertex)),
                     all.data(), GL_STREAM_DRAW);
        const GLsizei nLines = static_cast<GLsizei>(m_impl->lineVerts.size());
        const GLsizei nFills = static_cast<GLsizei>(m_impl->fillVerts.size());
        if (nLines > 0) { glDrawArrays(GL_LINES, 0, nLines); }
        if (nFills > 0) { glDrawArrays(GL_TRIANGLES, nLines, nFills); }
    }

    // ---- text ----
    if (!m_impl->textVerts.empty())
    {
        glUseProgram(m_impl->progText);
        glUniform1i(m_impl->locFontTex, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_impl->fontTex);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(m_impl->vaoText);
        glBindBuffer(GL_ARRAY_BUFFER, m_impl->vboText);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<trt_gl::GLsizeiptrT>(m_impl->textVerts.size() * sizeof(TextVertex)),
                     m_impl->textVerts.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_impl->textVerts.size()));
        glDisable(GL_BLEND);
    }

    glBindVertexArray(0);
    glfwSwapBuffers(m_impl->window);
}

}  // namespace trt_alpha::renderer
