// =============================================================================
//  trt_alpha :: datasource :: source_config
// -----------------------------------------------------------------------------
//  SourceConfig -- data-source configuration.
// =============================================================================
#pragma once

#include <string>

namespace trt_alpha::datasource {

//! Data-source type.
enum class SourceType
{
    Image,     //!< a single image
    Images,    //!< an image directory (scans the common formats)
    Video,     //!< a video file, or a network stream URL (rtsp / rtmp / http(s) / udp)
    Camera,    //!< USB / network camera
};

struct SourceConfig
{
    SourceType type = SourceType::Image;

    //! Image / directory / video path (relative to the project root, or
    //! absolute), or a stream URL. URLs are recognised by core::Paths::isUrl():
    //! they go straight to the FFmpeg backend and skip local file validation.
    //! Ignored when type == Camera.
    //! Note: a URL is always treated as a stream (type Image combined with a URL
    //! errors out explicitly), because cv::imread only understands local files.
    std::string path;

    //! Camera ID (valid when type == Camera).
    int cameraId = -1;

    //! Batch accumulation size (the engine batch size).
    int batchSize = 1;

    //! Whether to loop the video (valid when type == Video).
    //! Only applies to local video files -- a stream cannot be rewound.
    bool loop = false;

    //! Stream open / read timeouts (ms; only effective for URL streams on the
    //! FFmpeg/GStreamer backend). <= 0 = leave it to the backend default (i.e.
    //! wait forever; an unreachable stream blocks for a long time). Default 5s:
    //! enough slack for a slow stream, yet it will not hang forever.
    int openTimeoutMs = 5000;
    int readTimeoutMs = 5000;

    //! Data-source identifier (used to tell sources apart with multiple sources).
    int sourceId = -1;
};

//! The type (and the target it points at) deduced from one user-supplied source
//! string. See resolveSource() for the rule.
struct ResolvedSource
{
    SourceType  type = SourceType::Image;
    std::string path;                  //!< set unless type == Camera
    int         cameraId = -1;         //!< set when type == Camera
};

//! Interpret one "source" string the same way everywhere -- the CLI `--source`
//! and InferParams::source share this single rule, so the two can never drift:
//!   * a URL (rtsp / rtmp / http(s) / udp)  -> Video  (stream, FFmpeg backend)
//!   * all digits ("0", "1", ...)           -> Camera (device id)
//!   * a known video extension              -> Video  (local file)
//!   * anything else                        -> Image  (a directory upgrades itself
//!                                            to Images inside the source)
//! Empty input yields the default (Image / empty path); callers reject it.
[[nodiscard]] ResolvedSource resolveSource(const std::string& source);

}  // namespace trt_alpha::datasource
