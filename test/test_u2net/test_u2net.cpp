// =============================================================================
//  test/test_u2net/test_u2net.cpp
// -----------------------------------------------------------------------------
//  U2Net 测试：
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
    std::cout << "=== u2net tests ===\n";

    // [1]
    {
        bool ok = false;
        try
        {
            auto m = ModelRegistry::instance().create("u2net");
            ok = (m != nullptr) && (m->name() == "u2net");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] create threw: " << e.what() << "\n";
        }
        check(ok, "[1] 'u2net' registered and created");
    }

    // [2]
    {
        ModelConfig cfg;
        cfg.engine = "/definitely/not/exist/u2net_12345.trt";
        cfg.batchSize = 1;
        cfg.dstH = 320;
        cfg.dstW = 320;

        bool threw = false;
        try
        {
            auto m = ModelRegistry::instance().create("u2net");
            m->init(cfg);
        }
        catch (const std::runtime_error& e)
        {
            threw = true;
            std::cout << "       expected exception: " << e.what() << "\n";
        }
        check(threw, "[2] init with missing engine throws");
    }

    std::cout << "====================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}