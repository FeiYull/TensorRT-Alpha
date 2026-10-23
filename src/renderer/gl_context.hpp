// =============================================================================
//  gl_context -- GPU display context (private to the GL renderer)
// -----------------------------------------------------------------------------
//  Owns the GLFW window, the GL 3.3 core context, the shader programs and the
//  per-source textures. It is the only place that touches OpenGL / GLFW; the
//  renderer above it only speaks "upload a frame / draw a box / draw text".
//
//  Single-threaded by contract: the render thread owns it (Pipeline forces
//  render_threads = 1 whenever --show is on), so there is no locking here.
//
//  v1 uploads HOST BGR8 frames via glTexSubImage2D (a device -> texture path via
//  CUDA-GL interop is a follow-up; it needs `IRenderer::acceptsDevicePixels` to
//  let the host-copy policy keep frames on the device).
// =============================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace trt_alpha::renderer {

class GLContext
{
public:
    //! Font metrics shared with the renderer (so it can size the label badge):
    //! a 5x7 bitmap font, kGlyphCellPx frame-pixels per font cell, kGlyphAdvance
    //! cells per character (5 glyph columns + 1 spacing column).
    static constexpr int kGlyphCellPx = 2;
    static constexpr int kGlyphAdvance = 6;   // cells per character
    static constexpr int kGlyphRows    = 7;

    explicit GLContext(int numTiles);
    ~GLContext();

    GLContext(const GLContext&) = delete;
    GLContext& operator=(const GLContext&) = delete;

    //! Create the window + GL context + shaders + font atlas. Returns false on
    //! any failure (no GL 3.3, missing entry point, shader error).
    bool init(const std::string& title);

    //! True once the user asked to close the window (title bar / ESC / q).
    [[nodiscard]] bool shouldClose() const;

    //! Mark a source as ended: its tile shows a dark "NO SIGNAL" placeholder
    //! instead of the (frozen) last frame. Idempotent.
    void markNoSignal(int tile);

    //! Upload a tightly-packed BGR8 host frame into tile `tile`'s texture.
    void uploadFrame(int tile, const std::uint8_t* data, int w, int h, int stride);

    //! Begin a new frame: poll events, refresh the letterbox layout from the
    //! current framebuffer, and clear the primitive batch buffers. Call before
    //! the drawRect / drawLine / drawText calls that build the frame.
    void beginFrame();

    //! Draw a filled axis-aligned rect (box edge / label badge / keypoint), in
    //! frame pixel coordinates. Color is RGB 0..255.
    void drawRect(int tile, float x0, float y0, float x1, float y1,
                  std::uint8_t r, std::uint8_t g, std::uint8_t b);

    //! Draw a 1-px line (skeleton / oriented-box outline), frame pixel coords.
    void drawLine(int tile, float x0, float y0, float x1, float y1,
                  std::uint8_t r, std::uint8_t g, std::uint8_t b);

    //! Draw a text string whose top-left sits at frame pixel (x0, y0).
    void drawText(int tile, float x0, float y0, const std::string& text,
                  std::uint8_t r, std::uint8_t g, std::uint8_t b);

    //! Present the whole grid and swap buffers (polls events, so it also drives
    //! ESC / q / window-close handling).
    void present();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace trt_alpha::renderer
