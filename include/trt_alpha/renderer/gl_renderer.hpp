// =============================================================================
//  trt_alpha :: renderer :: gl_renderer
// -----------------------------------------------------------------------------
//  GLRenderer -- the OpenGL implementation of IRenderer (GPU display).
//
//    * drawResult() collects boxes / labels / keypoints as primitives (in frame
//      pixel coordinates); it does NOT write into result.views (the pixels stay
//      raw, so no renderedBuffer is produced).
//    * show() uploads the frame of the result's source and draws its tile plus
//      the collected primitives into a grid window (GPU rasterization), one tile
//      per source (sourceId -> tile).
//    * save() is NOT implemented (disk output is a follow-up: nvEnc). The CLI
//      rejects `--render gl --save` up front, so this only trips on misuse.
//
//  Frames are uploaded by whichever path their memory space dictates:
//    * HOST   frames -> glTexSubImage2D (BGR8 -> RGBA8 texture)
//    * DEVICE frames -> CUDA-GL interop (a kernel expands BGR8 -> RGBA8 into a
//      mapped PBO, then glTexSubImage2D DMAs it into the texture; no D2H). This
//      is what acceptsDevicePixels() == true enables: the host-copy policy can
//      keep NVDEC frames on the device because the GL backend can display them
//      directly.
//
//  Thread safety: single render thread owns the GL context (Pipeline forces
//  render_threads = 1 when --show is on).
// =============================================================================
#pragma once

#include "trt_alpha/renderer/i_renderer.hpp"

#include <memory>

namespace trt_alpha::renderer {

class GLRenderer final : public IRenderer
{
public:
    //! @param numTiles  number of display tiles == number of sources
    explicit GLRenderer(int numTiles);
    ~GLRenderer() override;

    GLRenderer(const GLRenderer&) = delete;
    GLRenderer& operator=(const GLRenderer&) = delete;

    void drawResult(core::BatchResult& result,
                    const std::vector<core::ClassInfo>& classNames,
                    const RenderOptions& options) const override;

    void save(const core::BatchResult& result,
              const std::string& outputDir) const override;

    void show(const core::BatchResult& result,
              const std::string& windowName) const override;

    bool shouldClose() const noexcept override;
    void onSourceEnded(int sourceId) override;
    void pump() const override;
    bool acceptsDevicePixels() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace trt_alpha::renderer
