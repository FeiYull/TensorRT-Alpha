// =============================================================================
//  trt_alpha :: datasource :: opencv_source
// -----------------------------------------------------------------------------
//  OpenCVSource —— IDataSource 的 OpenCV 实现。
//
//  一个类支持四种源（Image / Images / Video / Camera），由 SourceConfig 决定。
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
    //! 构造：打开资源。失败抛 std::runtime_error。
    explicit OpenCVSource(const SourceConfig& cfg);
    ~OpenCVSource() override;

    //! 读下一批。返回 false 表示"没有更多了"。
    [[nodiscard]] bool next(core::Batch& out) override;

    //! 请求停止（线程安全）。
    void requestStop() override;

    [[nodiscard]] const char* typeName() const noexcept override;

private:
    // ---- 打开 ----
    void openImage();
    void openImages();
    void openVideo();
    void openCamera();

    // ---- 读一批 ----
    bool readOneImage(cv::Mat& out);      // 图片 / 目录
    bool readOneFrame(cv::Mat& out);      // 视频 / 摄像头

    //! 把 cv::Mat 填到 Batch 的第 i 帧。
    void copyIntoBatch(core::Batch& batch, int index, const cv::Mat& img);

    //! "下一帧"的输出文件名主干（不含扩展名），由渲染层直接拼 .jpg：
    //!   * 图片 = 原文件名（bus.jpg -> "bus"）
    //!   * 视频 = 源文件主干 + 帧号（demo.mp4 -> "demo_000123"）
    //!   * 流(URL) = URL 末段主干 + 帧号（rtsp://cam/live -> "live_000123"）
    //!   * 摄像头 = cam<id> + 帧号（"cam0_000123"）
    //! frameIndex 用绝对帧号（批内第 i 帧传 firstFrameIndex + i），保证视频/相机唯一。
    //! 无下一帧（图片读完）返回空串，渲染层回退 frame_<帧号>。
    [[nodiscard]] std::string nameForNextFrame(std::uint64_t frameIndex) const;

   

    SourceConfig m_cfg;

    // 图片目录模式：所有图片路径
    std::vector<std::string> m_imagePaths;
    std::size_t m_imageIndex = 0;

    // 视频 / 摄像头 / 网络流
    cv::VideoCapture m_capture;

    //! 是否网络流（URL）：流不做本地路径校验、不能 loop 回绕
    bool m_isStream = false;

    // 帧号
    std::uint64_t m_nextFrameIndex = 0;

    // 停止标志
    std::atomic<bool> m_stopRequested{false};
};

}  // namespace trt_alpha::datasource