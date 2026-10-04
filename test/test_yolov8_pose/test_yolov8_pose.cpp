// =============================================================================
//  test/test_yolov8_pose/test_yolov8_pose.cpp
// -----------------------------------------------------------------------------
//  YoloV8Pose tests:
//    [1] registry can create
//    [2] init fails (engine not exist) throws
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
    std::cout << "=== yolov8_pose tests ===\n";

    // [1]
    {
        bool ok = false;
        try
        {
            auto m = ModelRegistry::instance().create("yolov8_pose");
            ok = (m != nullptr) && (m->name() == "yolov8_pose");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] create threw: " << e.what() << "\n";
        }
        check(ok, "[1] 'yolov8_pose' registered and created");
    }

    // [2]
    {
        ModelConfig cfg;
        cfg.engine = "/definitely/not/exist/yolov8n-pose_12345.trt";
        cfg.batchSize = 1;
        cfg.dstH = 640;
        cfg.dstW = 640;

        bool threw = false;
        try
        {
            auto m = ModelRegistry::instance().create("yolov8_pose");
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