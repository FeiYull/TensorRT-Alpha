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

}  // namespace trt_alpha::datasource
