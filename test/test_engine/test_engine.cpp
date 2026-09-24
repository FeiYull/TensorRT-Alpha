// =============================================================================
//  test/test_engine/test_engine.cpp
// -----------------------------------------------------------------------------
//  TrtEngine 测试（不需要真引擎文件）：
//    [1] TensorDesc::volume 计算（静态 / 含动态）
//    [2] 加载不存在的文件抛异常
//    [3] 加载空文件抛异常
//    [4] 加载内容非法的文件抛异常
//    [5] buildFromOnnx 抛 logic_error（未实现）
//
//  可选 C：命令行传引擎路径时，测真实加载
//    用法：test_engine [<engine.trt>]
//    有参数时：测成功加载 + 列出 io tensors
// =============================================================================
#include "trt_alpha/core/engine.hpp"

#include <NvInfer.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using trt_alpha::core::DataType;
using trt_alpha::core::TensorDesc;
using trt_alpha::core::TrtEngine;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

fs::path writeTempFile(const std::string& name, const std::string& content)
{
    const fs::path p = fs::temp_directory_path() / name;
    std::ofstream out(p, std::ios::binary);
    out << content;
    return p;
}

void removeTempFile(const fs::path& p)
{
    std::error_code ec;
    fs::remove(p, ec);
}

}  // namespace

int main(int argc, char** argv)
{
    std::cout << "=== TrtEngine tests ===\n";

    // ---------------------------------------------------------------
    // [1] TensorDesc::volume
    // ---------------------------------------------------------------
    {
        TensorDesc d;
        d.shape.nbDims = 4;
        d.shape.d[0] = 1; d.shape.d[1] = 3; d.shape.d[2] = 640; d.shape.d[3] = 640;
        check(d.volume() == static_cast<std::size_t>(1) * 3 * 640 * 640,
              "[1a] static volume 1x3x640x640");

        TensorDesc dyn;
        dyn.shape.nbDims = 4;
        dyn.shape.d[0] = -1;   // 动态 batch
        dyn.shape.d[1] = 3;
        dyn.shape.d[2] = 640;
        dyn.shape.d[3] = 640;
        // 动态维视为 0（跳过） -> 3*640*640 = 1228800
        check(dyn.volume() == static_cast<std::size_t>(3) * 640 * 640,
              "[1b] dynamic batch treated as 0 (skipped)");

        TensorDesc scalar;
        scalar.shape.nbDims = 0;
        check(scalar.volume() == 1, "[1c] scalar volume == 1");
    }

    // ---------------------------------------------------------------
    // [2] 加载不存在的文件
    // ---------------------------------------------------------------
    {
        bool threw = false;
        try
        {
            (void)TrtEngine{"/definitely/not/exist/engine_12345.trt"};
        }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[2] nonexistent engine file throws");
    }

    // ---------------------------------------------------------------
    // [3] 加载空文件
    // ---------------------------------------------------------------
    {
        const fs::path p = writeTempFile("test_engine_empty.trt", "");
        bool threw = false;
        try
        {
            (void)TrtEngine{p.string()};
        }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[3] empty engine file throws");
        removeTempFile(p);
    }

    // ---------------------------------------------------------------
    // [4] 加载内容非法的文件
    // ---------------------------------------------------------------
    {
        const fs::path p = writeTempFile("test_engine_garbage.trt",
                                         "this is not a valid engine file");
        bool threw = false;
        try
        {
            (void)TrtEngine{p.string()};
        }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[4] garbage engine file throws");
        removeTempFile(p);
    }

    // ---------------------------------------------------------------
    // [5] buildFromOnnx 抛未实现
    // ---------------------------------------------------------------
    {
        bool threw = false;
        try
        {
            TrtEngine::buildFromOnnx("a.onnx", "a.trt");
        }
        catch (const std::logic_error&) { threw = true; }
        check(threw, "[5] buildFromOnnx throws logic_error (not implemented)");
    }

    // ---------------------------------------------------------------
    // 可选 C：命令行传引擎路径时，测真实加载
    // ---------------------------------------------------------------
    if (argc >= 2)
    {
        std::cout << "\n--- optional: load real engine ---\n";
        const std::string enginePath = argv[1];
        try
        {
            TrtEngine eng(enginePath);
            std::cout << "  engine loaded OK\n";
            std::cout << "  io tensors:\n";
            for (const auto& t : eng.ioTensors())
            {
                std::cout << "    " << (t.isInput ? "in " : "out")
                          << " '" << t.name << "' "
                          << trt_alpha::core::nameOf(t.dtype);
                std::cout << " shape=[";
                for (int i = 0; i < t.shape.nbDims; ++i)
                {
                    std::cout << t.shape.d[i]
                              << (i + 1 < t.shape.nbDims ? "," : "");
                }
                std::cout << "]\n";
            }
        }
        catch (const std::exception& e)
        {
            std::cout << "  [WARN] failed to load engine: " << e.what() << "\n";
        }
    }
    else
    {
        std::cout << "\n(no engine path provided; optional real-load test skipped)\n";
    }

    std::cout << "=======================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}