// =============================================================================
//  test/test_efficientdet/test_efficientdet.cpp
// -----------------------------------------------------------------------------
//  EfficientDet 测试：
//    [1] 注册中心能创建
//    [2] init 失败（engine 不存在）抛异常
// =============================================================================
#include "trt_alpha/core/model_registry.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using trt_alpha::ModelRegistry;
using trt_alpha::core::ModelConfig;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

}  // namespace

int main(int argc, char** argv)
{
    std::cout << "=== efficientdet tests ===\n";

    // [1] 注册中心能创建
    {
        bool ok = false;
        try
        {
            auto m = ModelRegistry::instance().create("efficientdet");
            ok = (m != nullptr) && (m->name() == "efficientdet");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] create threw: " << e.what() << "\n";
        }
        check(ok, "[1] 'efficientdet' registered and created");
    }

    // [2] init 失败（engine 不存在）
    {
        ModelConfig cfg;
        cfg.engine = "/definitely/not/exist/efficientdet_12345.trt";
        cfg.batchSize = 1;
        cfg.dstH = 512;
        cfg.dstW = 512;

        bool threw = false;
        try
        {
            auto m = ModelRegistry::instance().create("efficientdet");
            m->init(cfg);
        }
        catch (const std::runtime_error& e)
        {
            threw = true;
            std::cout << "       expected exception: " << e.what() << "\n";
        }
        check(threw, "[2] init with missing engine throws");
    }

    // [3] 真推理（需要 engine + 图片）
    if (argc >= 3)
    {
        std::cout << "\n--- real inference ---\n";
        const std::string enginePath = argv[1];
        const std::string imagePath = argv[2];

        try
        {
            ModelConfig cfg;
            cfg.engine = enginePath;
            cfg.batchSize = 1;
            cfg.dstH = 512;
            cfg.dstW = 512;
            cfg.extras["num_class"] = "91";
            cfg.extras["conf_thresh"] = "0.45";

            auto m = ModelRegistry::instance().create("efficientdet");
            m->init(cfg);

            // 这里只验证 init 成功；真推理通过 sample 跑
            check(true, "[3] init with real engine OK");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [3] exception: " << e.what() << "\n";
            ++g_failures;
        }
    }
    else
    {
        std::cout << "\n(no engine+image args; real inference test skipped)\n";
        std::cout << "  usage: test_efficientdet <engine.trt> <image>\n";
    }

    std::cout << "====================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}