// =============================================================================
//  trt_alpha :: datasource :: source_config（实现）
// -----------------------------------------------------------------------------
//  resolveSource -- one rule for turning a user-supplied source string into a
//  concrete (type, target) pair. Shared by the CLI `--source` and the high-level
//  InferParams.source so the two entry points stay in lock-step.
// =============================================================================
#include "trt_alpha/datasource/source_config.hpp"

#include "trt_alpha/core/paths.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace trt_alpha::datasource {
namespace {

//! True when the string is non-empty and made of decimal digits only.
bool allDigits(const std::string& s)
{
    return !s.empty() &&
           std::all_of(s.begin(), s.end(),
                       [](unsigned char c) { return std::isdigit(c) != 0; });
}

//! True when the path ends in a known video-file extension.
//! The query string / fragment is stripped first so a URL such as
//! "clip.mp4?token=..." is still recognised (and isUrl() catches it anyway).
bool hasVideoExt(const std::string& src)
{
    const std::string path = src.substr(0, src.find_first_of("?#"));
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos)
    {
        return false;
    }

    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    static const char* const kExts[] = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
        ".ts", ".flv", ".mpg", ".mpeg", ".wmv",
    };
    for (const char* e : kExts)
    {
        if (ext == e)
        {
            return true;
        }
    }
    return false;
}

}  // namespace

ResolvedSource resolveSource(const std::string& source)
{
    ResolvedSource rs;
    if (source.empty())
    {
        return rs;   // default: Image + empty path; callers treat this as "no source"
    }

    // A URL is always a stream: cv::imread only understands local files, so it has
    // to go to the FFmpeg backend as a Video source.
    if (core::Paths::isUrl(source))
    {
        rs.type = SourceType::Video;
        rs.path = source;
        return rs;
    }

    // Bare digits = camera device id (same intuition as cv::VideoCapture(0)).
    // An out-of-int-range number is not a meaningful device id, so fall through
    // and let it be handled as a path.
    if (allDigits(source))
    {
        try
        {
            rs.type     = SourceType::Camera;
            rs.cameraId = std::stoi(source);
            return rs;
        }
        catch (const std::exception&)
        {
            // falls through to the image path
        }
    }

    if (hasVideoExt(source))
    {
        rs.type = SourceType::Video;
        rs.path = source;
        return rs;
    }

    // Everything else is an image path. "Is it a directory?" is left to the source
    // itself, which switches to Images on its own (OpenCVSource::openImage), so no
    // filesystem probe is needed here.
    rs.type = SourceType::Image;
    rs.path = source;
    return rs;
}

// =============================================================================
//  Backend / hardware-acceleration names
// -----------------------------------------------------------------------------
//  One parser for the CLI (`--decode`, `--hwaccel`) and one for the high-level
//  API, so the two spellings can never disagree. parse*() are the lenient
//  variants (unknown -> Auto); require*() are what the entry points use, because
//  a typo must not silently change which backend runs.
// =============================================================================
const char* toString(DecodeBackend b) noexcept
{
    switch (b)
    {
    case DecodeBackend::Auto:   return "auto";
    case DecodeBackend::OpenCV: return "opencv";
    case DecodeBackend::FFmpeg: return "ffmpeg";
    }
    return "auto";
}

const char* toString(HwAccel h) noexcept
{
    switch (h)
    {
    case HwAccel::Auto:  return "auto";
    case HwAccel::None:  return "none";
    case HwAccel::Nvdec: return "nvdec";
    }
    return "auto";
}

const char* toString(HostCopy h) noexcept
{
    switch (h)
    {
    case HostCopy::Auto: return "auto";
    case HostCopy::On:   return "on";
    case HostCopy::Off:  return "off";
    }
    return "auto";
}

namespace {

//! Lower-case a copy of the string (ASCII only; the keywords are ASCII).
std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

DecodeBackend parseDecodeBackend(const std::string& s) noexcept
{
    const std::string k = lower(s);
    if (k == "opencv" || k == "cv" || k == "opencvsource") { return DecodeBackend::OpenCV; }
    if (k == "ffmpeg" || k == "av" || k == "avcodec")      { return DecodeBackend::FFmpeg; }
    return DecodeBackend::Auto;
}

HwAccel parseHwAccel(const std::string& s) noexcept
{
    const std::string k = lower(s);
    if (k == "none" || k == "off" || k == "cpu" || k == "sw") { return HwAccel::None; }
    if (k == "nvdec" || k == "cuda" || k == "cuvid" || k == "hw") { return HwAccel::Nvdec; }
    return HwAccel::Auto;
}

HostCopy parseHostCopy(const std::string& s) noexcept
{
    const std::string k = lower(s);
    // "off" must be tested before the bool spellings: it is also a bool-ish word
    // a user might type expecting "no copy", and the enum has no ambiguity here.
    if (k == "off" || k == "false" || k == "no" || k == "0" || k == "never")
    {
        return HostCopy::Off;
    }
    if (k == "on" || k == "true" || k == "yes" || k == "1" || k == "always")
    {
        return HostCopy::On;
    }
    return HostCopy::Auto;
}

DecodeBackend requireDecodeBackend(const std::string& s)
{
    const std::string k = lower(s);
    if (k.empty() || k == "auto") { return DecodeBackend::Auto; }

    const DecodeBackend b = parseDecodeBackend(k);
    if (b == DecodeBackend::Auto)
    {
        throw std::invalid_argument("unknown decode backend '" + s +
                                    "' (expected auto | opencv | ffmpeg)");
    }
    return b;
}

HwAccel requireHwAccel(const std::string& s)
{
    const std::string k = lower(s);
    if (k.empty() || k == "auto") { return HwAccel::Auto; }

    const HwAccel h = parseHwAccel(k);
    if (h == HwAccel::Auto)
    {
        throw std::invalid_argument("unknown hardware acceleration '" + s +
                                    "' (expected auto | none | nvdec)");
    }
    return h;
}

HostCopy requireHostCopy(const std::string& s)
{
    const std::string k = lower(s);
    if (k.empty() || k == "auto") { return HostCopy::Auto; }

    const HostCopy h = parseHostCopy(k);
    if (h == HostCopy::Auto)
    {
        throw std::invalid_argument("unknown need_host_copy value '" + s +
                                    "' (expected auto | on | off)");
    }
    return h;
}

}  // namespace trt_alpha::datasource
