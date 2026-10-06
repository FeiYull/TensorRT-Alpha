// =============================================================================
//  test/test_yunet/test_yunet.cpp
// -----------------------------------------------------------------------------
//  YuNet 测试：
//    [1] 注册中心能创建
//    [2] init 失败（engine 不存在）抛异常
//    [3] 拿非 YuNet 引擎 init 必须报错（不会静默跑）—— 需要 engines/yolov8n.trt
// =============================================================================
#include "trt_alpha/core/model_registry.hpp"

#include <fstream>
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
    std::cout << "=== yunet tests ===\n";

    // [1]
    {
        bool ok = false;
        try
        {
            auto m = ModelRegistry::instance().create("yunet");
            ok = (m != nullptr) && (m->name() == "yunet");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] create threw: " << e.what() << "\n";
        }
        check(ok, "[1] 'yunet' registered and created");
    }

    // [2]
    {
        ModelConfig cfg;
        cfg.engine = "/definitely/not/exist/yunet_12345.trt";
        cfg.batchSize = 1;
        cfg.dstH = 320;
        cfg.dstW = 320;

        bool threw = false;
        try
        {
            auto m = ModelRegistry::instance().create("yunet");
            m->init(cfg);
        }
        catch (const std::runtime_error& e)
        {
            threw = true;
            std::cout << "       expected exception: " << e.what() << "\n";
        }
        check(threw, "[2] init with missing engine throws");
    }

    // [3] 非 YuNet 引擎（I/O 名字对不上）必须被 I/O 守门拦下
    {
        const std::string probe = "engines/yolov8n.trt";
        std::ifstream f(probe, std::ios::binary);
        if (!f.good())
        {
            std::cout << "[SKIP] [3] " << probe << " not found (cwd?)\n";
        }
        else
        {
            ModelConfig cfg;
            cfg.engine = probe;
            cfg.batchSize = 1;
            cfg.dstH = 320;
            cfg.dstW = 320;

            bool threw = false;
            std::string msg;
            try
            {
                auto m = ModelRegistry::instance().create("yunet");
                m->init(cfg);
            }
            catch (const std::runtime_error& e)
            {
                threw = true;
                msg = e.what();
            }
            std::cout << "       " << msg << "\n";
            check(threw, "[3] non-YuNet engine rejected (no loc/conf/iou tensors)");
        }
    }

    std::cout << "====================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}