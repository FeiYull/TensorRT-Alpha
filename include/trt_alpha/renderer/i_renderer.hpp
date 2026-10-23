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

//! 渲染展示选项（纯 POD，只影响"怎么画"，不影响推理结果）。
//!
//! 这些是【按模型】的参数：pose 的关键点阈值、seg 的掩码透明度，不同任务默认值
//! 不同，放 base 会让任务间隐式耦合。它们由调用方从 <model>.ini 的 [render] 节
//! 读取后传入；渲染器仍然不知道具体模型，只是消费一份选项。
//!
//! 默认值 == 上游 ultralytics 的口径（Annotator.kpts 的 conf_thres、
//! Annotator.masks 的 alpha）。
struct RenderOptions
{
    float kptConfThresh = 0.25f;   //!< 关键点置信度阈值：低于此值不画点 / 不连骨架
    float maskAlpha = 0.5f;        //!< 实例分割掩码混合 alpha
};

class IRenderer
{
public:
    virtual ~IRenderer() = default;

    IRenderer(const IRenderer&) = delete;
    IRenderer& operator=(const IRenderer&) = delete;

    //! Draw a whole batch of results (in place, back into result.views' memory).
    //! It walks result.views[0..validCount-1] and dispatches automatically
    //! according to detections / segmentations / obbs / classifications /
    //! semantics / depths / keypoints (only what the model produced is filled).
    //! classNames is the array of "class name + colour" (from
    //! ModelConfig.classNames). options carries the per-model rendering knobs
    //! (keypoint threshold / mask alpha / ...) read from <model>.ini [render].
    virtual void drawResult(core::BatchResult& result,
                            const std::vector<core::ClassInfo>& classNames,
                            const RenderOptions& options) const = 0;

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

    // ---- display lifecycle (default no-op) ----------------------------------
    // These concern the *window*, not the drawing. The OpenCV backend keeps the
    // no-op defaults (imshow has no persistent window state to report); only a
    // backend that owns a window (the GL backend) overrides them.

    //! True once the user asked to close the display window (title-bar X / ESC /
    //! q). The pipeline polls it after each show() and stops the run.
    virtual bool shouldClose() const noexcept { return false; }

    //! A source ended (video EOF / image done). The backend should show a
    //! "no signal" state for that source's tile instead of its last frame.
    virtual void onSourceEnded(int sourceId) { (void)sourceId; }

    //! Drive the display when there is no new frame to show: poll events, keep
    //! the window responsive (so shouldClose() flips), and redraw if needed.
    //! The pipeline calls it after the last result while the window stays open.
    virtual void pump() const {}

    // ---- optional capabilities (default no-op) ------------------------------
    // These are configuration, not drawing: they are set once before start()
    // and only read afterwards. Backends that do not write files (or do not
    // care) simply keep the no-op default, so adding a backend never requires
    // implementing them.

    //! Register an input source (file or directory, already resolved). save()
    //! must refuse to write over these paths -- the last gate for
    //! "output == input", which no data source or call path can escape.
    virtual void addInputSource(const std::string& resolvedPath)
    {
        (void)resolvedPath;
    }

    //! Quality used by save() when the output format is JPEG (1..100).
    //! Out-of-range values are clamped and reported, never applied silently.
    virtual void setJpegQuality(int quality)
    {
        (void)quality;
    }

protected:
    IRenderer() = default;
};

}  // namespace trt_alpha::renderer
