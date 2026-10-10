// =============================================================================
//  trt_alpha :: renderer :: i_renderer
// -----------------------------------------------------------------------------
//  IRenderer -- the rendering abstraction.
//
//  Design principles:
//    * Knows only the result structs (BatchResult / Detection / Segmentation /
//      ClassScore), never a concrete model
//    * Swapping the backend (OpenCV -> Qt / Skia / JSON) means adding an
//      implementation and switching the pointer at the call site
//    * [Does not depend on OpenCV]: the interface never exposes cv::Mat; it uses
//      only core's types
//    * It only "draws / saves / shows"; it never "reads images / infers /
//      schedules"
//
//  Call-sequence contract:
//    drawResult() -> save() / show()
//    Draw first, then save / show. drawResult() modifies in place the memory
//    that result.views points to.
//
//  Thread safety:
//    * each implementation is responsible for its own thread safety
//    * typically a single render thread calls it serially, so no extra
//      synchronization is needed
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/class_info.hpp"

#include <string>
#include <vector>

namespace trt_alpha::renderer {

class IRenderer
{
public:
    virtual ~IRenderer() = default;

    IRenderer(const IRenderer&) = delete;
    IRenderer& operator=(const IRenderer&) = delete;

    //! Draw a whole batch of results (in place, back into result.views' memory).
    //! It walks result.views[0..validCount-1] and dispatches automatically
    //! according to detections / segmentations / classifications.
    //! classNames is the array of "class name + colour" (from
    //! ModelConfig.classNames).
    virtual void drawResult(core::BatchResult& result,
                            const std::vector<core::ClassInfo>& classNames) const = 0;

    //! Save: every valid image goes to <outputDir>/<frame source name>.jpg
    //!   * image source = the original filename (data/bus.jpg -> <outputDir>/bus.jpg)
    //!   * video / camera = <stem>_<frame index> (demo_000123.jpg / cam0_000123.jpg)
    //!   * falls back to frame_<index> when result.frameNames is missing or empty
    //!   * a file of the same name is overwritten directly, after a WARN is logged
    //! outputDir is created automatically when absent.
    virtual void save(const core::BatchResult& result,
                      const std::string& outputDir) const = 0;

    //! Show: cv::imshow + cv::waitKey(1) (non-blocking); only the first valid
    //! image is displayed. Use this for real-time "one image per frame". For
    //! batch scenarios use save().
    virtual void show(const core::BatchResult& result,
                      const std::string& windowName) const = 0;

protected:
    IRenderer() = default;
};

}  // namespace trt_alpha::renderer
