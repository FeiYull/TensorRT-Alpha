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
//  Frames reach the texture through one of two paths:
//    * host BGR8  -> glTexSubImage2D (GL expands RGB -> RGBA, alpha = 1.0)
//    * device BGR8 -> CUDA-GL interop: a kernel writes into a mapped PBO
//      (pixel-unpack buffer), then glTexSubImage2D DMAs the PBO into the texture
//      -- both on the GPU, no D2H copy.
//  Both store (B, G, R, alpha) in an RGBA8 texture so a single `.bgr` fragment
//  shader serves both. The device path only engages when the host-copy policy
//  kept frames on the device (see IRenderer::acceptsDevicePixels).
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

    //! Index of the zoomed tile, or -1 while the full grid is shown. The render
    //! loop reads this so it only emits primitives for the zoomed source; clicks
    //! inside the window flip it (see the mouse callback registered in init()).
    [[nodiscard]] int focusedTile() const;

    //! Probe whether CUDA-GL interop is possible on this machine, using a
    //! throwaway HIDDEN GL context that is destroyed before returning. It must
    //! NOT reuse the render thread's context: handing a context across threads
    //! breaks CUDA-GL interop on Windows (WGL ties interop to the creating
    //! thread's context). False on software rasterizers (WSL/WSLg hands out
    //! Mesa's llvmpipe), where the device upload path is impossible and the
    //! host-copy policy must fall back to a D2H copy + glTexSubImage2D upload.
    static bool probeInterop();

    //! Result of the probe above (cached). False on software rasterizers, where
    //! device-resident frames cannot reach a texture and the caller must hand
    //! over host pixels instead.
    [[nodiscard]] bool supportsDeviceFrames() const;

    //! Upload a tightly-packed BGR8 host frame into tile `tile`'s texture.
    void uploadFrame(int tile, const std::uint8_t* data, int w, int h, int stride);

    //! Upload a tightly-packed BGR8 DEVICE frame into tile `tile`'s texture via
    //! CUDA-GL interop (zero copy): a kernel expands BGR8 -> RGBA8 into a mapped
    //! PBO (pixel-unpack buffer), then glTexSubImage2D DMAs the PBO into the
    //! texture -- no D2H. `devData` is a device pointer with `stride` bytes per
    //! row (== w * 3 for the decoder's tight layout). The PBO is registered /
    //! mapped / unmapped lazily and cached across frames.
    void uploadFrameDevice(int tile, const std::uint8_t* devData, int w, int h, int stride);

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
