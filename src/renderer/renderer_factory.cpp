// =============================================================================
//  trt_alpha :: renderer :: renderer_factory（实现）
// -----------------------------------------------------------------------------
//  See renderer_factory.hpp for the selection rule. Only this file names
//  OpenCVRenderer (and, later, GLRenderer), so adding or removing a backend
//  never touches a call site.
// =============================================================================
#include "trt_alpha/renderer/renderer_factory.hpp"

#include "trt_alpha/core/logger.hpp"

#if defined(TRT_ALPHA_WITH_OPENCV) && TRT_ALPHA_WITH_OPENCV
#  include "trt_alpha/renderer/opencv_renderer.hpp"
#endif
#if defined(TRT_ALPHA_WITH_GL) && TRT_ALPHA_WITH_GL
#  include "trt_alpha/renderer/gl_renderer.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace trt_alpha::renderer {

bool opencvBackendAvailable() noexcept
{
#if defined(TRT_ALPHA_WITH_OPENCV) && TRT_ALPHA_WITH_OPENCV
    return true;
#else
    return false;
#endif
}

bool glBackendAvailable() noexcept
{
#if defined(TRT_ALPHA_WITH_GL) && TRT_ALPHA_WITH_GL
    return true;
#else
    return false;
#endif
}

Backend requireBackend(const std::string& name)
{
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s.empty() || s == "auto")   { return Backend::Auto; }
    if (s == "opencv")              { return Backend::OpenCV; }
    if (s == "gl" || s == "opengl") { return Backend::GL; }
    throw std::runtime_error("run: unknown render backend '" + name +
                             "' (expect auto | opencv | gl)");
}

const char* toString(Backend backend) noexcept
{
    switch (backend)
    {
    case Backend::Auto:   return "auto";
    case Backend::OpenCV: return "opencv";
    case Backend::GL:     return "gl";
    }
    return "unknown";
}

std::unique_ptr<IRenderer> makeRenderer(const RendererConfig& cfg)
{
    Backend want = cfg.backend;
    if (want == Backend::Auto)
    {
        if (opencvBackendAvailable())     { want = Backend::OpenCV; }
        else if (glBackendAvailable())    { want = Backend::GL; }
    }

    if (want == Backend::OpenCV)
    {
#if defined(TRT_ALPHA_WITH_OPENCV) && TRT_ALPHA_WITH_OPENCV
        auto r = std::make_unique<OpenCVRenderer>();
        for (const auto& p : cfg.inputPaths)
        {
            if (!p.empty()) { r->addInputSource(p); }
        }
        r->setJpegQuality(cfg.jpegQuality);
        TRT_LOG_INFO("makeRenderer: backend=opencv, protected inputs="
                     << cfg.inputPaths.size() << ", jpeg_quality=" << cfg.jpegQuality);
        return r;
#else
        throw std::runtime_error(
            "makeRenderer: the OpenCV backend was requested, but this build has none "
            "(TRT_ALPHA_WITH_OPENCV=OFF). Reconfigure with -DTRT_ALPHA_WITH_OPENCV=ON.");
#endif
    }

    if (want == Backend::GL)
    {
#if defined(TRT_ALPHA_WITH_GL) && TRT_ALPHA_WITH_GL
        auto r = std::make_unique<GLRenderer>(cfg.numTiles);
        for (const auto& p : cfg.inputPaths)
        {
            if (!p.empty()) { r->addInputSource(p); }
        }
        TRT_LOG_INFO("makeRenderer: backend=gl, tiles=" << cfg.numTiles
                     << ", protected inputs=" << cfg.inputPaths.size());
        return r;
#else
        throw std::runtime_error(
            "makeRenderer: the OpenGL backend was requested, but this build has none "
            "(TRT_ALPHA_WITH_GL=OFF). Reconfigure with -DTRT_ALPHA_WITH_GL=ON "
            "-DGLFW_ROOT=<glfw root>.");
#endif
    }

    throw std::runtime_error(
        "makeRenderer: no renderer backend is compiled in. "
        "Rebuild with -DTRT_ALPHA_WITH_OPENCV=ON (or -DTRT_ALPHA_WITH_GL=ON).");
}

}  // namespace trt_alpha::renderer
