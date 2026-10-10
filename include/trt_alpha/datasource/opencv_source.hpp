// =============================================================================
//  trt_alpha :: datasource :: opencv_source
// -----------------------------------------------------------------------------
//  OpenCVSource -- the OpenCV implementation of IDataSource.
//
//  One class covers all four source kinds (Image / Images / Video / Camera),
//  decided by SourceConfig.
// =============================================================================
#pragma once

#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/datasource/source_config.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>

namespace trt_alpha::datasource {

class OpenCVSource final : public IDataSource
{
public:
    //! Constructor: open the resource. Throws std::runtime_error on failure.
    explicit OpenCVSource(const SourceConfig& cfg);
    ~OpenCVSource() override;

    //! Read the next batch. Returning false means "there is no more".
    [[nodiscard]] bool next(core::Batch& out) override;

    //! Request a stop (thread-safe).
    void requestStop() override;

    [[nodiscard]] const char* typeName() const noexcept override;

private:
    // ---- Open ----
    void openImage();
    void openImages();
    void openVideo();
    void openCamera();

    // ---- Read one batch ----
    bool readOneImage(cv::Mat& out);      // image / directory
    bool readOneFrame(cv::Mat& out);      // video / camera

    //! Copy a cv::Mat into frame i of the Batch.
    void copyIntoBatch(core::Batch& batch, int index, const cv::Mat& img);

    //! Output filename stem for the "next frame" (no extension); the render
    //! layer appends .jpg directly:
    //!   * image = the original filename (bus.jpg -> "bus")
    //!   * video = source stem + frame index (demo.mp4 -> "demo_000123")
    //!   * stream (URL) = last URL segment + frame index (rtsp://cam/live -> "live_000123")
    //!   * camera = cam<id> + frame index ("cam0_000123")
    //! frameIndex is an absolute index (frame i of a batch is passed
    //! firstFrameIndex + i) so video / camera names stay unique.
    //! Returns an empty string when there is no next frame (image list
    //! exhausted); the renderer then falls back to frame_<index>.
    [[nodiscard]] std::string nameForNextFrame(std::uint64_t frameIndex) const;


    SourceConfig m_cfg;

    // Image-directory mode: every image path
    std::vector<std::string> m_imagePaths;
    std::size_t m_imageIndex = 0;

    // Video / camera / network stream
    cv::VideoCapture m_capture;

    //! Whether this is a network stream (URL): streams skip local path
    //! validation and cannot be looped
    bool m_isStream = false;

    // Frame index
    std::uint64_t m_nextFrameIndex = 0;

    // Stop flag
    std::atomic<bool> m_stopRequested{false};
};

}  // namespace trt_alpha::datasource
