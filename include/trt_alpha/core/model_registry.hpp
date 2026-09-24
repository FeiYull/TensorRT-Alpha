// =============================================================================
//  trt_alpha :: core :: model_registry
// -----------------------------------------------------------------------------
//  ModelRegistry —— 模型注册中心（工厂的运行时字典）。
//
//  Meyers 单例规避静态初始化顺序问题；互斥锁保证并发注册/创建安全。
//
//  用法：
//    * 在模型 .cpp 末尾写一行 TRT_ALPHA_REGISTER_MODEL("yolov8", det::YoloV8)
//    * 调用方通过 ModelRegistry::instance().create("yolov8") 创建实例
//
//  【重要】注册体位于静态库目标内，可执行文件必须以 WHOLE_ARCHIVE 方式
//  链接模型库（CMake 已固化），否则链接器会丢弃未引用的 .obj、注册不会发生。
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

    //! 全局单例。
    static ModelRegistry& instance() noexcept;

    //! 注册（重名返回 false）。由 TRT_ALPHA_REGISTER_MODEL 宏调用。
    bool add(const std::string& name, Factory factory);

    //! 按名创建实例。未知名字抛 std::runtime_error（信息里列出全部已注册名）。
    [[nodiscard]] std::unique_ptr<IModel> create(const std::string& name) const;

    //! 列出全部已注册名。
    [[nodiscard]] std::vector<std::string> names() const;

private:
    ModelRegistry() = default;

    mutable std::mutex m_mutex;
    std::unordered_map<std::string, Factory> m_factories;
};

}  // namespace trt_alpha

// =============================================================================
//  注册宏
// =============================================================================
//  用法（在模型 .cpp 末尾写一行）：
//    TRT_ALPHA_REGISTER_MODEL("yolov8", trt_alpha::det::YoloV8)
//
//  注册器名由 __LINE__ 生成（ClassName 可能含 "::"，不能直接参与 token 拼接）。
//
//  注意：注册体位于静态库目标内，可执行文件必须以 WHOLE_ARCHIVE 方式链接
//  模型库，否则链接器会丢弃未引用的 .obj、注册不会发生。
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