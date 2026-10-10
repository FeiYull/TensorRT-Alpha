// =============================================================================
//  trt_alpha :: core :: model_registry
// -----------------------------------------------------------------------------
//  ModelRegistry -- the model registration centre (a runtime dictionary for
//  the factory).
//
//  A Meyers singleton sidesteps the static initialization order problem and a
//  mutex keeps concurrent registration / creation safe.
//
//  Usage:
//    * Add one line at the end of a model's .cpp:
//        TRT_ALPHA_REGISTER_MODEL("yolov8", det::YoloV8)
//    * Callers create instances with ModelRegistry::instance().create("yolov8")
//
//  [IMPORTANT] The registrars live inside a static library target, so the
//  executable must link the model library with WHOLE_ARCHIVE (already fixed in
//  CMake). Otherwise the linker discards unreferenced .obj files and no
//  registration takes place.
// =============================================================================
#pragma once

#include "trt_alpha/core/model.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_alpha {

class ModelRegistry
{
public:
    using Factory = std::unique_ptr<IModel> (*)();

    //! Global singleton.
    static ModelRegistry& instance() noexcept;

    //! Register a factory (returns false on a duplicate name).
    //! Called by the TRT_ALPHA_REGISTER_MODEL macro.
    bool add(const std::string& name, Factory factory);

    //! Create an instance by name. An unknown name throws std::runtime_error
    //! (the message lists every registered name).
    [[nodiscard]] std::unique_ptr<IModel> create(const std::string& name) const;

    //! List every registered name.
    [[nodiscard]] std::vector<std::string> names() const;

private:
    ModelRegistry() = default;

    mutable std::mutex m_mutex;
    std::unordered_map<std::string, Factory> m_factories;
};

}  // namespace trt_alpha

// =============================================================================
//  registration macro
// =============================================================================
//  Usage (one line at the end of a model's .cpp):
//    TRT_ALPHA_REGISTER_MODEL("yolov8", trt_alpha::det::YoloV8)
//
//  The registrar name is generated from __LINE__ (ClassName may contain "::",
//  so it cannot take part in token pasting directly).
//
//  Note: the registrar lives inside a static library target, so the executable
//  must link the model library with WHOLE_ARCHIVE; otherwise the linker
//  discards unreferenced .obj files and no registration takes place.
// =============================================================================
#define TRT_ALPHA_CONCAT_IMPL(a, b) a##b
#define TRT_ALPHA_CONCAT(a, b) TRT_ALPHA_CONCAT_IMPL(a, b)

#define TRT_ALPHA_REGISTER_MODEL(modelName, ClassName)                                \
    namespace                                                                         \
    {                                                                                 \
    struct TRT_ALPHA_CONCAT(TrtAlphaRegistrar, __LINE__) final                        \
    {                                                                                 \
        TRT_ALPHA_CONCAT(TrtAlphaRegistrar, __LINE__)()                               \
        {                                                                             \
            ::trt_alpha::ModelRegistry::instance().add(                               \
                modelName, []() -> std::unique_ptr<::trt_alpha::IModel> {             \
                    return std::unique_ptr<::trt_alpha::IModel>(new ClassName());      \
                });                                                                   \
        }                                                                             \
    };                                                                                \
    static const TRT_ALPHA_CONCAT(TrtAlphaRegistrar, __LINE__)                        \
        TRT_ALPHA_CONCAT(trtAlphaRegistrarInstance, __LINE__);                        \
    }