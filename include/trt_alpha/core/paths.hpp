// =============================================================================
//  trt_alpha :: core :: paths -- project root discovery + relative path resolution
// -----------------------------------------------------------------------------
//  Goal: let the program find the default resources under configs/ and data/
//  from [any working directory], with no need to cd to the project root and no
//  need to edit the path strings in the code.
//
//  Root discovery order (the first hit wins; the result is cached once):
//      1. --root <dir> (passed in explicitly; calls setOverride)
//      2. the TRT_ALPHA_ROOT environment variable
//      3. walk up from the executable's directory looking for a "root marker"
//      4. the compile-time TRT_ALPHA_ROOT_DIR (the source root written by CMake,
//         handy for out-of-tree builds)
//      5. walk up from the current working directory looking for a "root marker"
//      6. fallback = the current working directory
//
//  A "root marker" is a configs/ subdirectory, or a .trt_alpha_root file (the
//  latter is a manual marker for "ship only the exe + data" releases).
//
//  Portable across Windows / Linux: the executable is located with
//  GetModuleFileNameW and /proc/self/exe respectively; paths are always carried
//  as std::filesystem::path so that narrow-character streams cannot break on
//  non-ASCII paths.
// =============================================================================
#pragma once

#include <filesystem>
#include <string>

namespace trt_alpha::core {

//! Global path policy (purely static, no instances).
//! All state is initialized once on the first root() call and is read-only
//! afterwards (unless setOverride is called explicitly).
class Paths
{
public:
    //! Set the root directory explicitly (from --root).
    //! Must be called before the first root(); otherwise throws std::logic_error.
    //! An empty string clears the override (back to automatic discovery).
    static void setOverride(const std::string& dir);

    //! The project root (absolute). Discovered and cached on the first call.
    [[nodiscard]] static const std::filesystem::path& root();

    //! A description of where the root came from (for logs / errors, e.g.
    //! "executable location").
    [[nodiscard]] static std::string rootSource();

    //! string -> path.
    //! For non-ASCII input, two sources are accepted: the command line (cmd's
    //! ANSI code page) and configuration files (UTF-8). Whichever actually
    //! exists wins; if neither does, the native encoding is assumed.
    [[nodiscard]] static std::filesystem::path toPath(const std::string& text);

    //! A relative path is expanded against root(); an absolute path is returned
    //! unchanged; an empty string yields an empty path.
    [[nodiscard]] static std::filesystem::path resolve(const std::string& text);

    //! Whether this is a network URL (of the form <scheme>://<host>...), such as
    //! rtsp / rtmp / http(s) / udp. The test follows the RFC 3986 scheme shape
    //! (starts with a letter, then [A-Za-z0-9+.-], at least 2 characters), so
    //! Windows drive paths such as "D:/a.jpg" and "C://x" are not misread as URLs.
    //! Purpose: a URL is not a local file, so local checks such as requireFile /
    //! directory scanning must be skipped for it.
    [[nodiscard]] static bool isUrl(const std::string& text);

    //! resolve() plus an existence check. When missing, it throws a readable
    //! error carrying "where the root came from + how to fix it".
    //! role is used in the error message (e.g. "config file" / "input image").
    [[nodiscard]] static std::filesystem::path requireFile(const std::string& text,
                                                          const std::string& role);

    //! Turn an absolute path into a string convenient for printing / logging
    //! (native encoding on Windows).
    [[nodiscard]] static std::string toDisplay(const std::filesystem::path& path);

    //! Resolve the "results output directory". One rule: whoever passed a
    //! directory explicitly wins (used as-is, with no subdirectory appended);
    //! only when nobody did, fall back to <default root>/<model name> (default
    //! root = "save").
    //!   resolveSaveDir("out", "yolov8") -> "out"
    //!   resolveSaveDir("",    "yolov8") -> "save/yolov8"
    //!   resolveSaveDir("",    "")       -> "save"
    [[nodiscard]] static std::string resolveSaveDir(const std::string& dir,
                                                    const std::string& modelName);
};

}  // namespace trt_alpha::core
