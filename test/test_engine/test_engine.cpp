// =============================================================================
//  test/test_engine/test_engine.cpp
// -----------------------------------------------------------------------------
//  TrtEngine 测试（不需要真引擎文件）：
//    [1] TensorDesc::volume 计算（静态 / 含动态）
//    [2] 加载不存在的文件抛异常
//    [3] 加载空文件抛异常
//    [4] 加载内容非法的文件抛异常
//    [5] buildFromOnnx 抛 logic_error（未实现）
//    [6] resolveBatch：静态纠正 / 动态钳制 / 越下界报错 / 上界契约校验
//
//  可选 C：命令行传引擎路径时，测真实加载
//    用法：test_engine [<engine.trt>]
//    有参数时：测成功加载 + 列出 io tensors + [7] setInputShape 守卫
//    [7] 对第一个 input 张量验证 setInputShape：
//        静态引擎 → 形状一致放行、形状不符必须抛异常（杜绝静默越界）；
//        动态引擎 → profile 内的 min/max 形状必须被接受。
// =============================================================================
#include "trt_alpha/core/engine.hpp"

#include <NvInfer.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using trt_alpha::core::DataType;
using trt_alpha::core::TensorDesc;
using trt_alpha::core::TrtEngine;
using trt_alpha::core::resolveBatch;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

nvinfer1::Dims mkDims(std::initializer_list<int> vals)
{
    nvinfer1::Dims d{};
    int i = 0;
    for (int v : vals) { d.d[i++] = v; }
    d.nbDims = i;
    return d;
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
    // [6] resolveBatch：静态 / 动态语义统一
    // ---------------------------------------------------------------
    {
        // ---- 静态引擎：shape=[8,3,640,640]，无 profile ----
        TensorDesc st;
        st.name = "images";
        st.isInput = true;
        st.shape = mkDims({8, 3, 640, 640});
        check(!st.isDynamicBatch(), "[6a] static engine isDynamicBatch == false");
        {
            const auto br = st.batchRange();
            check(br.min == 8 && br.opt == 8 && br.max == 8,
                  "[6b] static batchRange == {8,8,8}");
        }
        {
            const auto rb = resolveBatch(st, 1, "[t]", 0);
            check(rb.batch == 8 && rb.corrected && !rb.isDynamic,
                  "[6c] static: requested 1 -> corrected to 8");
        }
        {
            const auto rb = resolveBatch(st, 8, "[t]", 0);
            check(rb.batch == 8 && !rb.corrected,
                  "[6d] static: requested 8 -> no correction");
        }

        // ---- 动态引擎：shape=[-1,3,640,640]，profile min/opt/max = 1/2/4 ----
        TensorDesc dy;
        dy.name = "images";
        dy.isInput = true;
        dy.shape   = mkDims({-1, 3, 640, 640});
        dy.minShape = mkDims({1, 3, 640, 640});
        dy.optShape = mkDims({2, 3, 640, 640});
        dy.maxShape = mkDims({4, 3, 640, 640});
        check(dy.isDynamicBatch(), "[6e] dynamic engine isDynamicBatch == true");
        {
            const auto br = dy.batchRange();
            check(br.min == 1 && br.opt == 2 && br.max == 4,
                  "[6f] dynamic batchRange == {1,2,4}");
        }
        {
            const auto rb = resolveBatch(dy, 2, "[t]", 0);
            check(rb.batch == 2 && !rb.corrected && rb.isDynamic,
                  "[6g] dynamic: requested 2 (in range) -> 2");
        }
        {
            const auto rb = resolveBatch(dy, 6, "[t]", 0);
            check(rb.batch == 4 && rb.corrected,
                  "[6h] dynamic: requested 6 > max -> clamped to 4");
        }
        {
            bool threw = false;
            try { (void)resolveBatch(dy, 0, "[t]", 0); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[6i] dynamic: requested 0 < min -> throws");
        }

        // ---- 上界契约校验 ----
        {
            bool ok = true;
            try { (void)resolveBatch(dy, 2, "[t]", 4); }   // 声明 == 引擎 max
            catch (...) { ok = false; }
            check(ok, "[6j] declared max 4 == engine max -> ok");
        }
        {
            bool threw = false;
            try { (void)resolveBatch(dy, 2, "[t]", 6); }   // 声明 != 引擎 max
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[6k] declared max 6 != engine max 4 -> throws");
        }
        {
            bool threw = false;
            try { (void)resolveBatch(st, 8, "[t]", 4); }   // 静态引擎上界契约不符
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[6l] declared max 4 != static fixed 8 -> throws");
        }
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

            // -----------------------------------------------------------
            // [7] setInputShape 守卫（真实引擎）
            // -----------------------------------------------------------
            const TensorDesc* in = nullptr;
            for (const auto& t : eng.ioTensors())
            {
                if (t.isInput) { in = &t; break; }
            }
            if (in != nullptr)
            {
                std::cout << "\n--- [7] setInputShape guard on '" << in->name << "' ---\n";
                if (!in->isDynamicBatch())
                {
                    // 静态引擎：形状由引擎写死。
                    bool ok = true;
                    try { eng.setInputShape(in->name, in->shape); }
                    catch (...) { ok = false; }
                    check(ok, "[7a] static: setInputShape(engine shape) -> ok");

                    nvinfer1::Dims badBatch = in->shape;
                    badBatch.d[0] = in->shape.d[0] + 1;
                    bool threwBatch = false;
                    try { eng.setInputShape(in->name, badBatch); }
                    catch (const std::runtime_error&) { threwBatch = true; }
                    check(threwBatch, "[7b] static: batch mismatch -> throws");

                    if (in->shape.nbDims >= 3)
                    {
                        nvinfer1::Dims badSp = in->shape;
                        badSp.d[in->shape.nbDims - 1] =
                            in->shape.d[in->shape.nbDims - 1] + 2;
                        bool threwSp = false;
                        try { eng.setInputShape(in->name, badSp); }
                        catch (const std::runtime_error&) { threwSp = true; }
                        check(threwSp, "[7c] static: spatial mismatch -> throws");
                    }

                    // 真实静态引擎的 TensorDesc 喂进 resolveBatch：
                    // 请求值 != 固定值 → 应纠正为引擎固定 batch。
                    const auto rb = resolveBatch(*in, in->shape.d[0] + 2, "[t]", 0);
                    check(rb.batch == in->shape.d[0] && rb.corrected && !rb.isDynamic,
                          "[7d] static: resolveBatch(requested+2) -> corrected to engine batch");
                }
                else if (in->minShape.nbDims == in->shape.nbDims)
                {
                    // 动态引擎：profile 内的形状应被接受。
                    bool okMin = true;
                    try { eng.setInputShape(in->name, in->minShape); }
                    catch (...) { okMin = false; }
                    check(okMin, "[7a] dynamic: setInputShape(profile min) -> ok");

                    bool okMax = true;
                    try { eng.setInputShape(in->name, in->maxShape); }
                    catch (...) { okMax = false; }
                    check(okMax, "[7b] dynamic: setInputShape(profile max) -> ok");

                    try { eng.setInputShape(in->name, in->minShape); } catch (...) {}
                }
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