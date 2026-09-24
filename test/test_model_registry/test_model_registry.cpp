// =============================================================================
//  test/test_model_registry/test_model_registry.cpp
// -----------------------------------------------------------------------------
//  IModel + ModelRegistry 测试（用 FakeModel，不依赖 TRT）：
//    [1] FakeModel 实现 IModel 全部接口（编译通过）
//    [2] 注册 + 创建
//    [3] 重名注册返回 false
//    [4] 未知名创建抛异常
//    [5] names() 列出全部
//    [6] TRT_ALPHA_REGISTER_MODEL 宏生效（静态注册）
// =============================================================================
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using trt_alpha::IModel;
using trt_alpha::ModelRegistry;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

//! FakeModel —— 实现 IModel 全部接口，用来测注册中心。
class FakeModel final : public IModel
{
public:
    const std::string& name() const noexcept override
    {
        static const std::string kName = "fake";
        return kName;
    }

    void init(const trt_alpha::core::ModelConfig&) override {}
    void setBatch(const trt_alpha::core::Batch&) override {}
    void preprocess() override {}
    void infer() override {}
    void postprocess() override {}
    void commitResult(trt_alpha::core::BatchResult&) override {}
    void reset() override {}

    const std::vector<trt_alpha::core::TensorDesc>& describe() const noexcept override
    {
        static const std::vector<trt_alpha::core::TensorDesc> kEmpty;
        return kEmpty;
    }
};

//! 另一个 FakeModel 子类，用于静态注册测试。
class RegisteredFake final : public IModel
{
public:
    const std::string& name() const noexcept override
    {
        static const std::string kName = "registered_fake";
        return kName;
    }
    void init(const trt_alpha::core::ModelConfig&) override {}
    void setBatch(const trt_alpha::core::Batch&) override {}
    void preprocess() override {}
    void infer() override {}
    void postprocess() override {}
    void commitResult(trt_alpha::core::BatchResult&) override {}
    void reset() override {}
    const std::vector<trt_alpha::core::TensorDesc>& describe() const noexcept override
    {
        static const std::vector<trt_alpha::core::TensorDesc> kEmpty;
        return kEmpty;
    }
};

}  // namespace

// 静态注册（在 main 之前执行）
TRT_ALPHA_REGISTER_MODEL("registered_fake", RegisteredFake)

int main()
{
    std::cout << "=== ModelRegistry tests ===\n";

    // [1] FakeModel 能实例化（接口实现完整）
    {
        auto m = std::make_unique<FakeModel>();
        check(m != nullptr, "[1] FakeModel can be instantiated");
        check(m->name() == "fake", "[1] name() returns 'fake'");
    }

    // [2] 注册 + 创建
    {
        auto& reg = ModelRegistry::instance();
        const bool ok = reg.add("fake_test", []() -> std::unique_ptr<IModel> {
            return std::make_unique<FakeModel>();
        });
        check(ok, "[2] add 'fake_test' returns true");

        auto m = reg.create("fake_test");
        check(m != nullptr, "[2] create 'fake_test' non-null");
        check(m->name() == "fake", "[2] instance works");
    }

    // [3] 重名注册返回 false
    {
        auto& reg = ModelRegistry::instance();
        const bool ok = reg.add("fake_test", []() -> std::unique_ptr<IModel> {
            return std::make_unique<FakeModel>();
        });
        check(!ok, "[3] duplicate add returns false");
    }

    // [4] 未知名创建抛异常
    {
        bool threw = false;
        try
        {
            (void)ModelRegistry::instance().create("no_such_model_12345");
        }
        catch (const std::runtime_error& e)
        {
            threw = true;
            std::cout << "       exception: " << e.what() << "\n";
        }
        check(threw, "[4] unknown name throws runtime_error");
    }

    // [5] names() 列出全部
    {
        auto names = ModelRegistry::instance().names();
        bool hasFakeTest = false;
        bool hasRegisteredFake = false;
        for (const auto& n : names)
        {
            if (n == "fake_test")       { hasFakeTest = true; }
            if (n == "registered_fake") { hasRegisteredFake = true; }
        }
        check(hasFakeTest, "[5] names() contains 'fake_test'");
        check(hasRegisteredFake, "[5] names() contains 'registered_fake' (static reg)");
    }

    std::cout << "====================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}