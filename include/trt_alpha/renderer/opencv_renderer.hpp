// =============================================================================
//  trt_alpha :: renderer :: opencv_renderer
// -----------------------------------------------------------------------------
//  OpenCVRenderer -- the OpenCV implementation of IRenderer.
//
//  Rendering style (fixed internally, not exposed to users):
//    * box line width 2, font FONT_HERSHEY_DUPLEX, font scale 0.5
//    * colours cycle by label (the same label always gets the same colour)
//    * masks are blended per pixel (alpha = 0.35)
//
//  Safety conventions:
//    * every ROI access is clipped to the image bounds first (no out-of-range)
//    * a mask of the wrong type is skipped rather than crashing the process
//    * save() refuses to write to an "input source file" -- an output path equal
//      to an input file would overwrite the original
// =============================================================================
#pragma once

#include "trt_alpha/renderer/i_renderer.hpp"

#include <string>
#include <unordered_set>

namespace trt_alpha::renderer {

class OpenCVRenderer final : public IRenderer
{
public:
    OpenCVRenderer() = default;
    ~OpenCVRenderer() override = default;

    //! Register an input source (a file or a directory, absolute path). A
    //! directory is expanded into the files inside it. save() refuses to write
    //! when it hits these paths -- this is the last gate for "output == input",
    //! which no data source and no call path (CLI / Infer) can escape. Must be
    //! called before start() (save() only reads it, so save can stay const).
    void addInputSource(const std::string& resolvedPath);

    [[nodiscard]] bool protectsInputs() const noexcept { return !m_inputPaths.empty(); }

    void drawResult(core::BatchResult& result,
                    const std::vector<core::ClassInfo>& classNames) const override;

    void save(const core::BatchResult& result,
              const std::string& outputDir) const override;

    void show(const core::BatchResult& result,
              const std::string& windowName) const override;

private:
    //! The set of normalised input file paths (see pathKey in the cpp).
    std::unordered_set<std::string> m_inputPaths;
};

}  // namespace trt_alpha::renderer
