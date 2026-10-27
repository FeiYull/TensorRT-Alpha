// =============================================================================
//  overlay_stage -- bakes task overlays into a frame, on the GPU
// -----------------------------------------------------------------------------
//  Renderer-private (like gl_context.hpp). It owns the CUDA-side staging:
//
//    host BGR frame  --H2D-->  device frame  --overlay kernels-->  device frame
//
//  The GL backend feeds the device frame straight into its PBO (CUDA-GL
//  interop), so no D2H happens at all; the OpenCV backend asks for the host copy
//  back. Either way there is exactly ONE implementation of the overlay maths --
//  `kernels/ops/overlay.cu` -- and the old per-pixel host painting in
//  `opencv_renderer.cpp` is kept for reference but no longer called.
//
//  Single-threaded by contract: the render thread owns it.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "trt_alpha/core/batch_result.hpp"

namespace trt_alpha::renderer {

class OverlayStage
{
public:
    OverlayStage();
    ~OverlayStage();

    OverlayStage(const OverlayStage&) = delete;
    OverlayStage& operator=(const OverlayStage&) = delete;

    //! Bake every overlay carried by `result` into a DEVICE copy of `hostBgr`.
    //! Returns the device frame (valid until the next call), or nullptr when the
    //! result carries no overlay at all (the caller then uploads `hostBgr` as is).
    const std::uint8_t* composeDevice(const core::BatchResult& result,
                                      const std::uint8_t* hostBgr,
                                      int width,
                                      int height,
                                      std::size_t stride,
                                      float maskAlpha,
                                      int index = 0);

    //! Copy the frame produced by the last successful composeDevice() back to
    //! the host (pinned staging). Returns nullptr if nothing was composed. Used
    //! by backends that upload textures from CPU memory because CUDA-GL interop
    //! is unavailable.
    const std::uint8_t* copyBack(int width, int height, std::size_t stride);

    //! Last CUDA error seen, for a single WARN instead of a silent no-op.
    [[nodiscard]] const char* lastError() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace trt_alpha::renderer
