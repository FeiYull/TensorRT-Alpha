// =============================================================================
//  trt_alpha :: datasource :: source_factory
// -----------------------------------------------------------------------------
//  The single place that turns (SourceConfig.decodeBackend) into a concrete
//  IDataSource. Everything above it -- the CLI, InferParams, every sample --
//  calls makeSource() and never names a concrete source class again, so adding
//  or removing a backend is a one-file change.
//
//  Rule (see makeSource):
//      Image / Images / Camera  -> OpenCV (FFmpeg is a video decoder here)
//      Video (file or URL)      -> what cfg.decodeBackend asks for
//      Auto                     -> FFmpeg when this build has it, else OpenCV
// =============================================================================
#pragma once

#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/datasource/source_config.hpp"

#include <memory>

namespace trt_alpha::datasource {

//! True when this binary was compiled with FFmpeg support
//! (TRT_ALPHA_WITH_FFMPEG) and the shared libraries are present.
[[nodiscard]] bool ffmpegBackendAvailable() noexcept;

//! Create the data source cfg asks for.
//! Throws when cfg.decodeBackend == FFmpeg but this build has no FFmpeg support:
//! silently downgrading would make a benchmark compare the wrong two things.
[[nodiscard]] std::unique_ptr<IDataSource> makeSource(const SourceConfig& cfg);

//! Scan a raw argv for `--decode <opencv|ffmpeg|auto>` and
//! `--hwaccel <auto|none|nvdec>` and apply them to cfg. Unknown flags are
//! ignored (they belong to the caller). Samples use this so none of them has to
//! re-implement the two switches.
void applyBackendArgs(int argc, char* const argv[], SourceConfig& cfg);

}  // namespace trt_alpha::datasource
