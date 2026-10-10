// =============================================================================
//  trt_alpha :: core :: model_config
// -----------------------------------------------------------------------------
//  ModelConfig -- runtime configuration for a model.
//
//  Design:
//    * Common fields (shared by every model): engine / batchSize / dstH / dstW /
//      inputOutputNames / classNamesFile / classNames
//    * Model-specific fields: all live in extras (unordered_map<string, string>)
//      and are read by the model itself via getInt / getFloat / getString /
//      getBool
//
//  Why extras instead of "one Config subclass per model":
//    * IModel::init keeps its const ModelConfig& signature, so no dynamic_cast
//    * Adding a model does not touch ModelConfig
//    * A config is fundamentally "runtime strings", so type safety is enforced
//      at read time
//
//  Sources:
//    * INI file (configs/<model>.ini)
//    * CLI arguments, which override some fields
//    * classNames, read from the class file at runtime
// =============================================================================
#pragma once

#include "trt_alpha/core/class_info.hpp"
#include "trt_alpha/core/layout.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <memory>

namespace trt_alpha::core {
    class Engine;   // forward declaration
}

namespace trt_alpha::core {

struct ModelConfig
{
    // ---- Fields shared by every model ----
    std::string engine;                        //!< engine path (relative to the repo root, or absolute)
    std::string classNamesFile;                //!< path of the class file (name + RGB)
    int batchSize = 1;                         //!< batch actually used for this inference (corrected at runtime to the engine's capability)
    int maxBatchSize = -1;                     //!< optional: declared upper-bound contract for the engine batch; <= 0 means undeclared
    //! Logical axis order of the input (INI: input.layout, e.g. nchw / nhwc / ncdhw).
    //! Empty = use the model's own canonical layout. H/W no longer need any
    //! configuration: the shape declared by the engine is the single source of truth.
    Layout layout;
    //! "Intent" values for the spatial axes (INI: input.dst_h / input.dst_w).
    //! 0 = unset (default). Usually unnecessary: static axes always follow the
    //! engine-declared shape. Only when that axis is dynamic (-1) in the engine
    //! does this value take effect, as the intent for "how large to run".
    int dstH = 0;
    int dstW = 0;
    std::vector<std::string> inputOutputNames;

    // ---- Model-specific fields (all gathered here) ----
    // key = the original INI key (including the section prefix, e.g.
    //       "model.num_class") -- the section-stripped short name ("num_class")
    //       is stored as well
    // value = the original INI string value
    std::unordered_map<std::string, std::string> extras;

    //! Each key's [origin] + [first-seen order] (long names only, e.g.
    //! "model.engine"). source is one of "base.ini" / "<model>.ini" / "CLI".
    //! Order = the order in base.ini; keys added by the model ini are appended
    //! afterwards (duplicate keys keep their first position).
    //! Used only for config display (logConfigBox); it takes part in no logic.
    std::vector<std::pair<std::string, std::string>> origins;

    //! Write / overwrite the origin of a long key (called when the CLI overrides
    //! the config). Keeps the first-seen order.
    void setOrigin(const std::string& fullKey, const std::string& source);

    //! Query the origin of a long key; returns `fallback` when not found.
    [[nodiscard]] std::string originOf(const std::string& fullKey,
                                       const std::string& fallback = "-") const;

    //! Records keys that were actually read (short or long name; whatever was
    //! read gets recorded).
    //!   * Registered automatically by getInt / getFloat / getString / getBool;
    //!   * loadModelConfig registers explicitly the keys it reads straight from
    //!     IniParser.
    //! Purpose: logConfigBox uses this to flag dead keys -- present in the ini
    //! but never read on the current path (typically a key under the wrong
    //! section, which silently has no effect). Pure diagnostics; it takes part
    //! in no logic.
    //! Note: reads happen during init (single-threaded per model) and are
    //! read-only afterwards, so no locking is needed.
    mutable std::unordered_set<std::string> readKeys;

    //! Mark a key as consumed (short or long name).
    void markRead(const std::string& key) const { readKeys.insert(key); }

    //! Whether a long key was consumed: either an exact hit or a hit on the
    //! section-stripped short name.
    [[nodiscard]] bool wasRead(const std::string& fullKey) const;

    // ---- Filled at runtime ----
    std::vector<ClassInfo> classNames;

    //! Shared engine (optional). When non-null, IModel::init reuses it instead
    //! of deserializing from the engine path again. Used for "1 engine + N
    //! contexts".
    std::shared_ptr<Engine> sharedEngine;

    // ---- Convenience readers (absent -> fallback; conversion failure -> throw) ----
    [[nodiscard]] std::string getString(const std::string& key,
                                       const std::string& fallback = "") const;
    [[nodiscard]] int getInt(const std::string& key, int fallback) const;
    [[nodiscard]] float getFloat(const std::string& key, float fallback) const;
    [[nodiscard]] bool getBool(const std::string& key, bool fallback) const;
};

}  // namespace trt_alpha::core
