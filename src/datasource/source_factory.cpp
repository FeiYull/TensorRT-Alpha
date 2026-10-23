// =============================================================================
//  trt_alpha :: datasource :: source_factory（实现）
// -----------------------------------------------------------------------------
//  The only place that names a concrete IDataSource. Every caller above it --
//  the CLI, InferParams, the 20 samples -- goes through makeSource(), so the
//  backend is chosen by configuration and nothing else.
//
//  See source_factory.hpp for the routing rule.
// =============================================================================
#include "trt_alpha/datasource/source_factory.hpp"

#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/datasource/ffmpeg_source.hpp"

#if defined(TRT_ALPHA_WITH_OPENCV) && TRT_ALPHA_WITH_OPENCV
#  include "trt_alpha/datasource/opencv_source.hpp"
#endif

#include <stdexcept>
#include <string>

namespace trt_alpha::datasource {
namespace {

const char* typeName(SourceType t) noexcept
{
    switch (t)
    {
    case SourceType::Image:  return "image";
    case SourceType::Images: return "images";
    case SourceType::Video:  return "video";
    case SourceType::Camera: return "camera";
    }
    return "unknown";
}

//! Build the OpenCV-backed source, or fail loudly in an OpenCV-less build.
//!
//! Every non-video source type still routes here: a per-OS capture backend for
//! cameras is out of scope, and image decoding without OpenCV is a separate
//! step. Failing loudly beats silently handing back FFmpegSource for a camera.
std::unique_ptr<IDataSource> makeOpenCVSource(const SourceConfig& cfg)
{
#if defined(TRT_ALPHA_WITH_OPENCV) && TRT_ALPHA_WITH_OPENCV
    return std::make_unique<OpenCVSource>(cfg);
#else
    throw std::runtime_error(
        std::string("makeSource: source type '") + typeName(cfg.type) +
        "' needs the OpenCV backend, but this build has none "
        "(TRT_ALPHA_WITH_OPENCV=OFF). Video files can still use the FFmpeg "
        "backend (-DTRT_ALPHA_WITH_FFMPEG=ON).");
#endif
}

}  // namespace

bool ffmpegBackendAvailable() noexcept
{
#if defined(TRT_ALPHA_WITH_FFMPEG) && TRT_ALPHA_WITH_FFMPEG
    return true;
#else
    return false;
#endif
}

std::unique_ptr<IDataSource> makeSource(const SourceConfig& cfg)
{
    // FFmpeg is a *video* decoder here: images, image directories and cameras
    // stay on OpenCV (a device-capture backend differs per OS for no gain).
    if (cfg.type != SourceType::Video)
    {
        if (cfg.decodeBackend == DecodeBackend::FFmpeg)
        {
            TRT_LOG_WARN("makeSource: the FFmpeg backend decodes video only, source type '"
                         << typeName(cfg.type) << "' stays on OpenCV");
        }
        return makeOpenCVSource(cfg);
    }

    DecodeBackend want = cfg.decodeBackend;
    if (want == DecodeBackend::Auto)
    {
        want = ffmpegBackendAvailable() ? DecodeBackend::FFmpeg : DecodeBackend::OpenCV;
    }

    if (want == DecodeBackend::FFmpeg)
    {
        // Asking for FFmpeg in a build without it must be loud: silently using
        // OpenCV would make a benchmark compare the wrong two things.
        if (!ffmpegBackendAvailable())
        {
            throw std::runtime_error(
                "makeSource: FFmpeg decoding was requested but this build has no FFmpeg "
                "support (TRT_ALPHA_WITH_FFMPEG=OFF). Reconfigure with "
                "-DTRT_ALPHA_WITH_FFMPEG=ON -DFFMPEG_ROOT=<shared ffmpeg build>, or use "
                "--decode opencv.");
        }
        return std::make_unique<FFmpegSource>(cfg);
    }

    return makeOpenCVSource(cfg);
}

void applyBackendArgs(int argc, char* const argv[], SourceConfig& cfg)
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        const std::string a = (argv[i] != nullptr) ? argv[i] : "";
        if (a != "--decode" && a != "--hwaccel")
        {
            continue;
        }
        const std::string v = (argv[i + 1] != nullptr) ? argv[i + 1] : "";
        if (a == "--decode")
        {
            cfg.decodeBackend = requireDecodeBackend(v);
        }
        else
        {
            cfg.hwAccel = requireHwAccel(v);
        }
        ++i;
    }
}

}  // namespace trt_alpha::datasource
