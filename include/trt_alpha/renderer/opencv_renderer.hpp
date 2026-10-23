// =============================================================================
//  trt_alpha :: renderer :: opencv_renderer
// -----------------------------------------------------------------------------
//  OpenCVRenderer -- the OpenCV implementation of IRenderer.
//
//  Rendering style (fixed internally, not exposed to users):
//    * box line width 2, font FONT_HERSHEY_DUPLEX, font scale 0.5
//    * colours cycle by label (the same label always gets the same colour)
//    * per-model knobs (mask alpha / keypoint threshold) come in via RenderOptions
//
//  Safety conventions:
//    * every ROI access is clipped to the image bounds first (no out-of-range)
//    * a mask of the wrong type is skipped rather than crashing the process
//    * save() refuses to write to an "input source file" -- an output path equal
//      to an input file would overwrite the original
// =============================================================================
#pragma once

#include "trt_alpha/renderer/i_renderer.hpp"

#include <mutex>
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
    void addInputSource(const std::string& resolvedPath) override;

    [[nodiscard]] bool protectsInputs() const noexcept { return !m_inputPaths.empty(); }

    //! JPEG quality used by save() -- 1..100, 95 == OpenCV's own default.
    //! Out-of-range values are clamped (and reported), never silently applied.
    //! Must be called before start(): save() only reads it, so save stays const
    //! and the several render threads need no extra locking.
    void setJpegQuality(int quality) override;

    void drawResult(core::BatchResult& result,
                    const std::vector<core::ClassInfo>& classNames,
                    const RenderOptions& options) const override;

    void save(const core::BatchResult& result,
              const std::string& outputDir) const override;

    void show(const core::BatchResult& result,
              const std::string& windowName) const override;

private:
    //! The set of normalised input file paths (see pathKey in the cpp).
    std::unordered_set<std::string> m_inputPaths;

    //! JPEG quality for save() (see setJpegQuality). Written once before start(),
    //! only read afterwards -- hence no locking around the const save().
    int m_jpegQuality = 95;

    //! Output directory cache. save() is called once per frame and is const, so
    //! the expensive part (create_directories + weakly_canonical) cannot be done
    //! in the constructor -- it is memoised on first use instead. save() may also
    //! be called by several render threads at once, hence mutable + a mutex.
    mutable std::mutex m_outDirMutex;
    mutable std::string m_outDirPath;      //!< the outputDir the cache was built for
    mutable std::string m_outDirCanon;     //!< ... and its normalised form
    mutable bool m_outDirReady = false;    //!< an attempt has been made
    mutable bool m_outDirOk = false;       //!< ... and it succeeded
    mutable bool m_outDirWarnedOverwrite = false;
};

}  // namespace trt_alpha::renderer
