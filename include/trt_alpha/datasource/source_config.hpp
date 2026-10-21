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

//! Which library reads / decodes the frames.
//!
//! The two backends are interchangeable: both produce tightly packed BGR8 host
//! frames, so nothing downstream (preprocess / infer / render) can tell them
//! apart. Picking one is a command-line / INI decision, not a code change.
enum class DecodeBackend
{
    Auto,     //!< FFmpeg when the build has it and the target needs it, else OpenCV
    OpenCV,   //!< cv::VideoCapture + cv::imread (always available)
    FFmpeg,   //!< libavformat + libavcodec; requires TRT_ALPHA_WITH_FFMPEG
};

//! How the FFmpeg backend decodes. Ignored by the OpenCV backend.
enum class HwAccel
{
    Auto,     //!< NVDEC when the codec + GPU support it, silent fallback to None
    None,     //!< libavcodec on the CPU (software decode)
    Nvdec,    //!< NVDEC (cuvid); hard error when it is not available
};

//! Whether a decoded frame is copied back to host memory.
//!
//! The host copy exists for exactly one consumer: the renderer, which draws into
//! those pixels. When nothing renders, the model reads the device buffer and the
//! pixels are never touched again -- on NVDEC that copy costs 0.475 ms of a
//! 1.70 ms 1080p frame. This enum is the user-facing switch; SourceConfig keeps
//! the resolved boolean.
enum class HostCopy
{
    Auto,   //!< copy only when something actually reads host pixels (default)
    On,     //!< always copy (e.g. the caller wants Frame::image())
    Off,    //!< never copy; the pixels stay on the device
};

[[nodiscard]] const char* toString(DecodeBackend b) noexcept;
[[nodiscard]] const char* toString(HwAccel h) noexcept;
[[nodiscard]] const char* toString(HostCopy h) noexcept;

//! Parse "opencv" / "ffmpeg" / "auto" (case-insensitive). Unknown -> Auto.
[[nodiscard]] DecodeBackend parseDecodeBackend(const std::string& s) noexcept;
//! Parse "auto" / "none" / "nvdec" (case-insensitive). Unknown -> Auto.
[[nodiscard]] HwAccel parseHwAccel(const std::string& s) noexcept;
//! Parse "auto" / "on" / "off" (also true/false/1/0). Unknown -> Auto.
[[nodiscard]] HostCopy parseHostCopy(const std::string& s) noexcept;

//! Same, but a typo throws instead of silently becoming "auto".
//!
//! Why this exists: "auto" is a value the user can spell, so it cannot double as
//! the "unknown input" sentinel -- "--decode ffmpg" would quietly pick FFmpeg
//! and the benchmark would compare something the user did not ask for.
//! "" and "auto" are still accepted as Auto.
[[nodiscard]] DecodeBackend requireDecodeBackend(const std::string& s);
[[nodiscard]] HwAccel requireHwAccel(const std::string& s);
[[nodiscard]] HostCopy requireHostCopy(const std::string& s);

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

    //! Which library decodes the frames (see DecodeBackend).
    //! Default OpenCV: the FFmpeg backend is opt-in, so adding it cannot change
    //! the behaviour (or the frame rate) of an existing command line.
    DecodeBackend decodeBackend = DecodeBackend::OpenCV;

    //! Decoder preference for the FFmpeg backend (see HwAccel).
    HwAccel hwAccel = HwAccel::Auto;

    //! Stream open / read timeouts (ms; only effective for URL streams on the
    //! FFmpeg backend). <= 0 = leave it to the backend default (i.e. wait
    //! forever; an unreachable stream blocks for a long time). Default 5s:
    //! enough slack for a slow stream, yet it will not hang forever.
    int openTimeoutMs = 5000;
    int readTimeoutMs = 5000;

    //! Whether the host copy of every decoded frame is actually needed.
    //!
    //! Resolved from the tri-state HostCopy, which the user sets through
    //! `[source] need_host_copy` in the ini, `--host-copy` on the CLI, or
    //! InferParams::need_host_copy. HostCopy::Auto keeps the default below.
    //!
    //! Only the renderer reads host pixels: it draws boxes into them and the
    //! result is then shown or written to disk. When neither `--show` nor
    //! `--save` is requested, that copy is pure overhead -- the model reads the
    //! device buffer, and the result is thrown away.
    //!
    //! On the NVDEC path the overhead is one 6.2 MB device-to-host copy per
    //! frame, measured at 0.475 ms of a 2.14 ms 1080p frame (22 %). Skipping it
    //! also removes the page-locked allocation and the completion event the
    //! batch would otherwise carry.
    //!
    //! Default true on purpose: the demos and the high-level Infer API hand
    //! batches to things that do read the pixels, and they must keep working
    //! without knowing this field exists. Sources that produce host pixels for
    //! other reasons (OpenCV) ignore it -- they have no device path to fall back
    //! to, so "no host copy" would mean "no pixels at all".
    bool needHostCopy = true;

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
