// =============================================================================
//  trt_alpha :: app :: commands（实现）
// =============================================================================
#include "commands.hpp"

#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <iostream>

namespace trt_alpha::app {

void printUsage()
{
    std::cout <<
        "trt_alpha v1.0\n"
        "\n"
        "Usage:\n"
        "  trt_alpha list\n"
        "  trt_alpha run    --image <path> [--engine <trt>] [--save] [--show]\n"
        "  trt_alpha run    --video <path> [--engine <trt>] [--save]\n"
        "  trt_alpha bench  (TODO)\n"
        "  trt_alpha build  (TODO)\n"
        "\n"
        "Options:\n"
        "  --config <ini>   model INI (default: configs/yolov8.ini)\n"
        "  --engine <trt>   override INI's engine path\n"
        "  --model <name>   model name (default: yolov8)\n"
        "  --batch <n>      override INI's batch_size\n"
        "  --save           save result images\n"
        "  --save-dir <dir> output dir (default: save)\n"
        "  --show           show result window\n"
        "  --workers <n>    inference pool workers (default: 0=auto)\n"
        "  --root <dir>     override project root\n";
}

int listCommand(const std::vector<std::string>& /*args*/)
{
    const auto names = trt_alpha::ModelRegistry::instance().names();

    std::cout << "Registered models (" << names.size() << "):\n";
    for (const auto& n : names)
    {
        std::cout << "  " << n << "\n";
    }

    return 0;
}

int runCommand(const std::vector<std::string>& /*args*/)
{
    TRT_LOG_WARN("trt_alpha run: not implemented yet");
    return 1;
}

int benchCommand(const std::vector<std::string>& /*args*/)
{
    TRT_LOG_WARN("trt_alpha bench: not implemented yet");
    return 1;
}

int buildCommand(const std::vector<std::string>& /*args*/)
{
    TRT_LOG_WARN("trt_alpha build: not implemented yet");
    return 1;
}

}  // namespace trt_alpha::app