// =============================================================================
//  trt_alpha :: core :: config
// -----------------------------------------------------------------------------
//  Loads ModelConfig from an INI file, and class information from a TXT file.
// =============================================================================
#pragma once

#include "trt_alpha/core/class_info.hpp"
#include "trt_alpha/core/model_config.hpp"

#include <string>
#include <vector>

namespace trt_alpha::core {

//! Forward declaration: logConfigBox only takes I/O descriptions by pointer,
//! so this header does not need engine.hpp (which would drag NvInfer.h into
//! every consumer).
struct TensorDesc;

//! INI -> ModelConfig.
//! engine / classNamesFile / inputOutputNames / numClass are mandatory;
//! a missing one throws std::runtime_error (the message carries context).
//! Relative paths are kept as-is (the caller decides what they are relative to).
//! Also fills cfg.origins: whether each key came from base.ini or from this
//! model's ini (order preserved).
[[nodiscard]] ModelConfig loadModelConfig(const std::string& iniPath);

//! TXT -> vector<ClassInfo>.
//! One line per entry: "name R G B" (RGB 0-255); the line number is the label.
//! A malformed line or an unreadable file throws std::runtime_error.
[[nodiscard]] std::vector<ClassInfo> loadClassNamesFile(const std::string& txtPath);

// -----------------------------------------------------------------------------
//  Config display: box up "the configuration actually in effect"
// -----------------------------------------------------------------------------
//! Print a config box (double rules top and bottom, single pipes left and
//! right); it is emitted in Release builds too.
//!   * Body: every key merged from base.ini + the model ini, grouped by section;
//!     each line reads "short-name = value  origin". Keys added to the ini show
//!     up automatically, with no code change needed.
//!   * [Actually effective]: a key that no consumer ever read is tagged
//!     `[unused]` -- decided by ModelConfig::readKeys (auto-registered by the
//!     getXxx accessors plus explicit registration in loadModelConfig).
//!   * Appended [resolved] section: engine truth (whether the batch was
//!     corrected, plus the shape, dtype and physical format of every input and
//!     output). Omitted when io == nullptr.
//!
//! @param cfg            a config already loaded, already overridden by the CLI,
//!                       and already read by its consumers. Prefer passing
//!                       InferencePool::modelConfig() (the very copy the model
//!                       uses, so readKeys is complete); passing an outer cfg
//!                       would mis-tag the model's keys as unused.
//! @param netName        model registration name (used in the title line)
//! @param iniPath        model ini path (used in the title line)
//! @param io             engine I/O tensor descriptions (usually
//!                       InferencePool::ioDesc(), valid only after the model's
//!                       init); nullptr = do not print the [resolved] section
//! @param resolvedBatch  batch after the engine corrected it; <= 0 = unknown
//!                       (that line is omitted)
void logConfigBox(const ModelConfig& cfg,
                  const std::string& netName,
                  const std::string& iniPath,
                  const std::vector<TensorDesc>* io = nullptr,
                  int resolvedBatch = 0);

}  // namespace trt_alpha::core
