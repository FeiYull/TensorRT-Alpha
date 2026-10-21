// =============================================================================
//  trt_alpha :: datasource :: ffmpeg_source
// -----------------------------------------------------------------------------
//  FFmpegSource -- the libavformat + libavcodec implementation of IDataSource.
//
//  Scope (deliberately narrow):
//      * Video files and network streams (rtsp / rtmp / http / udp).
//      * Images, image directories and cameras stay on OpenCVSource: the ask was
//        "choose OpenCV or FFmpeg for *video decode*", and a device-capture
//        backend differs per OS (dshow on Windows, v4l2 on Linux) for no gain.
//
//  What it produces is indistinguishable from OpenCVSource: tightly packed
//  BGR8 host frames, one core::Buffer per batch, views laid out exactly the
//  same way. preprocess / infer / postprocess / render cannot tell them apart.
//
//  Acceleration:
//      * HwAccel::Nvdec -- decode on NVDEC (h264_cuvid / hevc_cuvid / ...), then
//        convert NV12 -> BGR8 on the GPU and download once. The CPU never
//        touches a pixel.
//      * HwAccel::None  -- libavcodec software decode + libswscale.
//      * HwAccel::Auto  -- try NVDEC, fall back to software with a WARN.
//
//  This header includes no FFmpeg header on purpose (pImpl): the public surface
//  of trt_alpha must not drag libav* macros into every translation unit.
// =============================================================================
#pragma once

#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/datasource/source_config.hpp"

#include <memory>

namespace trt_alpha::datasource {

class FFmpegSource final : public IDataSource
{
public:
    //! Opens the media immediately; throws with a readable message on failure.
    //! Only SourceType::Video (local file or URL) is accepted.
    explicit FFmpegSource(const SourceConfig& cfg);
    ~FFmpegSource() override;

    FFmpegSource(const FFmpegSource&) = delete;
    FFmpegSource& operator=(const FFmpegSource&) = delete;

    [[nodiscard]] bool next(core::Batch& out) override;
    void requestStop() override;
    [[nodiscard]] const char* typeName() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace trt_alpha::datasource
