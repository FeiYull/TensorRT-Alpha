// =============================================================================
//  trt_alpha :: app :: 入口
// -----------------------------------------------------------------------------
//  分发命令：run / bench / list / build。
// =============================================================================
#include "commands.hpp"

#include "trt_alpha/core/logger.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    trt_alpha::core::installCrashHandler();

    try
    {
        std::vector<std::string> args;
        args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
        for (int i = 1; i < argc; ++i)
        {
            args.emplace_back(argv[i]);
        }

        if (args.empty())
        {
            trt_alpha::app::printUsage();
            return 0;
        }

        const std::string& command = args[0];

        if (command == "list")  { return trt_alpha::app::listCommand(args); }
        if (command == "run")   { return trt_alpha::app::runCommand(args); }
        if (command == "bench") { return trt_alpha::app::benchCommand(args); }
        if (command == "build") { return trt_alpha::app::buildCommand(args); }

        if (command == "help" || command == "--help" || command == "-h")
        {
            trt_alpha::app::printUsage();
            return 0;
        }

        std::cerr << "unknown command: " << command << "\n";
        trt_alpha::app::printUsage();
        return 1;
    }
    catch (const std::exception& e)
    {
        TRT_LOG_ERROR("trt_alpha: " << e.what());
        return 1;
    }
}