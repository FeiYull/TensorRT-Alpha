// =============================================================================
//  trt_alpha :: renderer :: renderer_factory
// -----------------------------------------------------------------------------
//  The only place above the implementations that names a concrete IRenderer.
//  Adding a backend means: add an enum value + an implementation + one branch
//  here. Callers never learn which one they got.
//
//  Backend selection (same shape as datasource::makeSource's DecodeBackend):
//    Auto   -> the first backend that this build actually compiled in.
//              Order: OpenCV, then GL. Auto is therefore always "whatever the
//              build supports", never a silent fallback to something broken.
//    OpenCV -> explicit; fails loudly in a build without OpenCV.
//    GL     -> explicit; fails loudly in a build without OpenGL.
//
//  "Fails loudly" is deliberate: silently handing back a different backend
//  makes an A/B comparison compare the wrong two things (the same rule that
//  makeSource() follows for the FFmpeg decoder).
// =============================================================================
#pragma once

#include "trt_alpha/renderer/i_renderer.hpp"

#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::renderer {

//! Which rendering backend to instantiate.
enum class Backend
{
    Auto = 0,   //!< first backend compiled into this build (OpenCV -> GL)
    OpenCV,     //!< cv::Mat drawing + cv::imshow + cv::imwrite
    GL          //!< OpenGL: CUDA-GL interop texture, GPU rasterization
};

//! Everything a backend may need at construction time.
//!
//! These used to be setters on the concrete class (addInputSource /
//! setJpegQuality). They moved here so the factory can hand a fully configured
//! object to a caller that only knows IRenderer -- and so that a future backend
//! cannot be added without deciding what it does about input protection.
struct RendererConfig
{
    Backend backend = Backend::Auto;

    //! Resolved input paths. save() must never write over these.
    //! Empty == nothing to protect (e.g. a URL source, or save disabled).
    std::vector<std::string> inputPaths;

    //! JPEG quality for save(), 1..100. Clamped by the implementation.
    int jpegQuality = 95;

    //! Number of display tiles == number of sources. Only the GL backend reads
    //! it (it lays the streams out as a grid); the OpenCV backend shows a single
    //! window per frame and ignores this. Default 1 = a single stream.
    int numTiles = 1;
};

//! True when this build compiled the OpenCV backend (TRT_ALPHA_WITH_OPENCV).
[[nodiscard]] bool opencvBackendAvailable() noexcept;

//! True when this build compiled the OpenGL backend (TRT_ALPHA_WITH_GL).
[[nodiscard]] bool glBackendAvailable() noexcept;

//! Parse "auto" / "opencv" / "gl" (case-insensitive). "" and "auto" -> Auto.
//! Unlike Backend::Auto itself, a typo throws (same rule as requireDecodeBackend:
//! "auto" is a value the user can spell, so it cannot double as the unknown sentinel).
[[nodiscard]] Backend requireBackend(const std::string& name);

//! Create a renderer. Throws std::runtime_error when the requested backend is
//! not part of this build (Auto included: a build with no backend at all is a
//! configuration error, not something to paper over).
[[nodiscard]] std::unique_ptr<IRenderer> makeRenderer(const RendererConfig& cfg);

//! Stable name for logs ("auto" / "opencv" / "gl").
[[nodiscard]] const char* toString(Backend backend) noexcept;

}  // namespace trt_alpha::renderer
