// =============================================================================
//  test/test_engine/test_engine.cpp
// -----------------------------------------------------------------------------
//  TrtEngine 测试（不需要真引擎文件）：
//    [1] TensorDesc::volume 计算（静态 / 含动态）
//    [2] 加载不存在的文件抛异常
//    [3] 加载空文件抛异常
//    [4] 加载内容非法的文件抛异常
//    [5] buildFromOnnx 抛 logic_error（未实现）
//    [6] resolveBatch：静态不符报错 / 动态越界报错 / 非法值 / 上界契约校验
//    [8] Layout：轴字母串解析（任意排列 / 任意秩 / 大小写 / 非法输入）
//    [9] validateInputTensor：秩 / 通道轴 / 物理格式 三重护栏
//    [10] resolveInputShape：静态维按引擎纠正、动态维取意图值、5D / NHWC / CHWN
//
//  可选 C：命令行传引擎路径时，测真实加载
//    用法：test_engine [<engine.trt>]
//    有参数时：测成功加载 + 列出 io tensors + [7] setInputShape 守卫
//              + [11] 物理格式必须线性 + [12] 真实引擎上的布局解析
//              + [13] applyInputShape 的 batch 护栏（bench / sample 走的路径）
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
using trt_alpha::core::Layout;
using trt_alpha::core::ResolvedInputShape;
using trt_alpha::core::TensorDesc;
using trt_alpha::core::ModelConfig;
using trt_alpha::core::TrtEngine;
using trt_alpha::core::applyInputShape;
using trt_alpha::core::resolveBatch;
using trt_alpha::core::resolveInputShape;
using trt_alpha::core::validateInputTensor;

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

//! 期望抛 std::runtime_error 的简写。
template <typename Fn>
bool throwsRuntime(Fn&& fn)
{
    try { fn(); }
    catch (const std::runtime_error&) { return true; }
    catch (...) { return false; }
    return false;
}

TensorDesc mkTensor(std::initializer_list<int> shape,
                    nvinfer1::TensorFormat fmt = nvinfer1::TensorFormat::kLINEAR)
{
    TensorDesc t;
    t.name    = "input";
    t.isInput = true;
    t.shape   = mkDims(shape);
    t.format  = fmt;
    return t;
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
            bool threw = false;
            try { (void)resolveBatch(st, 1, "[t]", 0); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[6c] static: requested 1 != fixed 8 -> throws");
        }
        {
            const auto rb = resolveBatch(st, 8, "[t]", 0);
            check(rb.batch == 8 && !rb.isDynamic,
                  "[6d] static: requested 8 == fixed 8 -> ok");
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
            check(rb.batch == 2 && rb.isDynamic,
                  "[6g] dynamic: requested 2 (in range) -> 2");
        }
        {
            bool threw = false;
            try { (void)resolveBatch(dy, 6, "[t]", 0); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[6h] dynamic: requested 6 > max -> throws");
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

        // ---- 非法请求值（ini 侧没有前置校验，必须在这里兜住）----
        {
            bool threwStatic = false;
            try { (void)resolveBatch(st, 0, "[t]", 0); }
            catch (const std::runtime_error&) { threwStatic = true; }
            check(threwStatic, "[6m] static: requested 0 -> throws");
        }
        {
            bool threwNeg = false;
            try { (void)resolveBatch(dy, -3, "[t]", 0); }
            catch (const std::runtime_error&) { threwNeg = true; }
            check(threwNeg, "[6n] dynamic: requested -3 -> throws");
        }
    }

    // ---------------------------------------------------------------
    // [8] Layout：轴字母串（任意排列 / 任意秩）
    // ---------------------------------------------------------------
    {
        check(Layout{"NCHW"}.rank() == 4 && Layout{"NCHW"}.indexOf('H') == 2 &&
              Layout{"NCHW"}.indexOf('W') == 3 && Layout{"NCHW"}.indexOf('C') == 1,
              "[8a] NCHW: rank 4, C@1 H@2 W@3");

        check(Layout{"NCHW"} == Layout{"nchw"}, "[8b] case-insensitive ('nchw' == 'NCHW')");

        {
            Layout out;
            check(!Layout::tryParse("", out), "[8c] empty string rejected");
        }
        {
            Layout out;
            check(!Layout::tryParse("NCHH", out), "[8d] duplicate axis letter rejected");
        }
        {
            Layout out;
            check(!Layout::tryParse("NCXW", out), "[8e] unknown axis letter rejected");
        }
        {
            Layout out;
            check(!Layout::tryParse("NCDHWTTWE", out), "[8f] rank > 8 rejected");
        }
        {
            Layout out;
            bool threw = false;
            try { (void)Layout{"NCHH"}; }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[8g] throwing ctor rejects invalid layout");
        }

        check(Layout::defaultForRank(3) == Layout{"CHW"} &&
              Layout::defaultForRank(4) == Layout{"NCHW"} &&
              Layout::defaultForRank(5) == Layout{"NCDHW"} &&
              Layout::defaultForRank(6).empty(),
              "[8h] defaultForRank: 3->CHW 4->NCHW 5->NCDHW 6->empty");

        check(Layout{"NCHW?"}.rank() == 5 && Layout{"NCHW?"}.has('?'),
              "[8i] wildcard '?' keeps rank and is repeatable");

        // 昇腾 CANN 真实存在的排列：CHWN
        const Layout chwn{"CHWN"};
        check(chwn.indexOf('N') == 3 && chwn.indexOf('H') == 1 && chwn.indexOf('W') == 2,
              "[8j] CHWN: N@3 H@1 W@2");

        // 5D
        check(Layout{"NDHWC"}.indexOf('D') == 1 && Layout{"NDHWC"}.indexOf('C') == 4,
              "[8k] NDHWC: D@1 C@4");
    }

    // ---------------------------------------------------------------
    // [9] validateInputTensor：秩 / 通道轴 / 物理格式 三重护栏
    // ---------------------------------------------------------------
    {
        // 合法：NCHW [1,3,640,640]
        check(!throwsRuntime([] { validateInputTensor(mkTensor({1, 3, 640, 640}),
                                                      Layout::NCHW, 3, "[t]"); }),
              "[9a] NCHW [1,3,640,640] ch=3 -> ok");

        // 合法：NHWC [1,512,512,3]
        check(!throwsRuntime([] { validateInputTensor(mkTensor({1, 512, 512, 3}),
                                                      Layout::NHWC, 3, "[t]"); }),
              "[9b] NHWC [1,512,512,3] ch=3 -> ok");

        // ① 秩不符
        check(throwsRuntime([] { validateInputTensor(mkTensor({1, 3, 640, 640}),
                                                     Layout::NCDHW, 3, "[t]"); }),
              "[9c] rank 4 vs layout NCDHW(rank 5) -> throws");

        // ② 通道轴不符（布局声明写错，绝不允许静默拿错轴）
        check(throwsRuntime([] { validateInputTensor(mkTensor({1, 4, 640, 640}),
                                                     Layout::NCHW, 3, "[t]"); }),
              "[9d] C-axis 4 != channels 3 -> throws");

        // NHWC 布局套在 NCHW 形状上：d[3]=640 != 3 → 立刻报错
        check(throwsRuntime([] { validateInputTensor(mkTensor({1, 3, 640, 640}),
                                                     Layout::NHWC, 3, "[t]"); }),
              "[9e] NHWC layout on NCHW shape -> throws");

        // ③ 物理格式必须线性
        check(throwsRuntime([] {
                  validateInputTensor(mkTensor({1, 3, 640, 640}, nvinfer1::TensorFormat::kCHW4),
                                      Layout::NCHW, 3, "[t]");
              }),
              "[9f] non-linear format (kCHW4) -> throws");

        // 空布局
        check(throwsRuntime([] { validateInputTensor(mkTensor({1, 3, 640, 640}),
                                                     Layout{}, 3, "[t]"); }),
              "[9g] empty layout -> throws");

        // channels <= 0 → 跳过通道校验（C=1 的模型 / 未知通道场景）
        check(!throwsRuntime([] { validateInputTensor(mkTensor({1, 1, 28, 28}),
                                                      Layout::NCHW, -1, "[t]"); }),
              "[9h] channels <= 0 skips channel check (C=1 grayscale ok)");
    }

    // ---------------------------------------------------------------
    // [10] resolveInputShape：引擎声明形状为唯一真相源
    // ---------------------------------------------------------------
    {
        ResolvedInputShape out;
        const ResolvedInputShape none;   // 无意图值

        // 静态 H/W + 错误意图值 → 按引擎纠正
        {
            ResolvedInputShape intent;
            intent.height = 640; intent.width = 640;
            resolveInputShape(mkTensor({1, 3, 512, 512}), Layout::NCHW, 3, intent, "[t]", out);
            check(out.height == 512 && out.width == 512 && out.corrected && !out.dynamic,
                  "[10a] static H/W: intent 640 -> corrected to engine 512");
        }
        // 静态 H/W + 一致意图值 → 不纠正
        {
            ResolvedInputShape intent;
            intent.height = 512; intent.width = 512;
            resolveInputShape(mkTensor({1, 3, 512, 512}), Layout::NCHW, 3, intent, "[t]", out);
            check(out.height == 512 && out.width == 512 && !out.corrected,
                  "[10b] static H/W: intent matches engine -> not corrected");
        }
        // 动态 H/W → 采用意图值
        {
            ResolvedInputShape intent;
            intent.height = 320; intent.width = 640;
            resolveInputShape(mkTensor({-1, 3, -1, -1}), Layout::NCHW, 3, intent, "[t]", out);
            check(out.height == 320 && out.width == 640 && out.dynamic && !out.corrected,
                  "[10c] dynamic H/W: intent 320x640 is used");
        }
        // NHWC
        {
            ResolvedInputShape intent;
            intent.height = 640; intent.width = 640;
            resolveInputShape(mkTensor({1, 512, 512, 3}), Layout::NHWC, 3, intent, "[t]", out);
            check(out.height == 512 && out.width == 512 && out.corrected,
                  "[10d] NHWC: H/W taken from d[1]/d[2], not d[2]/d[3]");
        }
        // 5D NCDHW
        {
            resolveInputShape(mkTensor({1, 3, 8, 224, 224}), Layout::NCDHW, 3, none, "[t]", out);
            check(out.depth == 8 && out.height == 224 && out.width == 224,
                  "[10e] NCDHW 5D: depth/height/width resolved");
        }
        // CHWN 排列：shape = [C,H,W,N]
        {
            resolveInputShape(mkTensor({3, 224, 224, 1}), Layout{"CHWN"}, 3, none, "[t]", out);
            check(out.height == 224 && out.width == 224,
                  "[10f] CHWN permutation: H/W resolved from d[1]/d[2]");
        }
        // 布局无 D 轴时，intent.depth 不该触发 corrected
        {
            ResolvedInputShape intent;
            intent.depth = 8;
            resolveInputShape(mkTensor({1, 3, 640, 640}), Layout::NCHW, 3, intent, "[t]", out);
            check(!out.corrected && out.depth == 0,
                  "[10g] no D axis in layout: intent.depth ignored");
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

            // 物理格式守卫：本框架只喂线性 buffer，非 kLINEAR 的引擎必须被拒绝
            {
                bool allLinear = true;
                for (const auto& t : eng.ioTensors())
                {
                    if (t.format != nvinfer1::TensorFormat::kLINEAR) { allLinear = false; }
                }
                check(allLinear, "[11a] all io tensors use kLINEAR");
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
                    // 请求值 != 固定值 → 抛异常（不静默纠正）。
                    bool threwRb = false;
                    try { (void)resolveBatch(*in, in->shape.d[0] + 2, "[t]", 0); }
                    catch (const std::runtime_error&) { threwRb = true; }
                    check(threwRb,
                          "[7d] static: resolveBatch(requested+2) -> throws");
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

            // -----------------------------------------------------------
            // [12] resolveInputShape / validateInputTensor（真实引擎）
            // -----------------------------------------------------------
            if (in != nullptr && in->shape.nbDims == 4)
            {
                // 用"通道数 == 3"识别真实布局：d[1]==3 → NCHW（yolo 系）；
                // d[3]==3 → NHWC（efficientdet）。两者都不是则跳过（如 mnist 的 C=1）。
                Layout l;
                if (in->shape.d[1] == 3)
                {
                    l = Layout::NCHW;
                }
                else if (in->shape.d[3] == 3)
                {
                    l = Layout::NHWC;
                }

                if (!l.empty())
                {
                    std::cout << "\n--- [12] layout resolution on '" << in->name
                              << "' as " << l.str() << " ---\n";
                    const int hIdx = l.indexOf('H');
                    const int wIdx = l.indexOf('W');

                    // 故意给一个错的 H/W 意图值：静态维必须被纠正为引擎声明值
                    ResolvedInputShape intent;
                    intent.height = 9999;
                    intent.width  = 9999;
                    ResolvedInputShape res;
                    resolveInputShape(*in, l, 3, intent, "[t]", res);

                    const int declaredH = in->shape.d[hIdx];
                    const int declaredW = in->shape.d[wIdx];
                    const int expectH = declaredH > 0 ? declaredH : 9999;   // 动态维保留意图值
                    const int expectW = declaredW > 0 ? declaredW : 9999;
                    check(res.height == expectH && res.width == expectW,
                          "[12a] real engine: H/W resolved from engine declaration");

                    // 声明成相反的布局必须报错（防止静默把 C 当 W）
                    const Layout wrong = (l == Layout::NCHW) ? Layout::NHWC : Layout::NCHW;
                    const bool wrongThrows = throwsRuntime([&] {
                        ResolvedInputShape r2;
                        resolveInputShape(*in, wrong, 3, {}, "[t]", r2);
                    });
                    check(wrongThrows,
                          "[12b] real engine: opposite layout declaration -> throws");

                    // 通道数对不上也必须报错
                    check(throwsRuntime([&] {
                              ResolvedInputShape r3;
                              resolveInputShape(*in, l, 7, {}, "[t]", r3);
                          }),
                          "[12c] real engine: wrong channel count -> throws");

                    // -------------------------------------------------------
                    // [13] applyInputShape 的 batch 护栏（真实引擎）
                    //   这是 bench / sample / 单测等"直连 model->init、不经过
                    //   InferencePool"路径的护栏；与池路径调同一个 resolveBatch，
                    //   报错措辞一致，不再退化成 TRT 的 satisfyProfile 原话。
                    // -------------------------------------------------------
                    std::cout << "\n--- [13] applyInputShape batch guard ---\n";
                    {
                        ModelConfig over;
                        over.batchSize = in->batchRange().max + 2;   // 必越界
                        check(throwsRuntime([&] {
                                  applyInputShape(eng, in->name, l, 3, over);
                              }),
                              "[13a] applyInputShape: batch over max -> throws");
                    }
                    {
                        ModelConfig zero;
                        zero.batchSize = 0;                          // ini 写 0 / 负数
                        check(throwsRuntime([&] {
                                  applyInputShape(eng, in->name, l, 3, zero);
                              }),
                              "[13b] applyInputShape: batch <= 0 -> throws");
                    }
                    {
                        // 合法值必须放行，且写回值 == 请求值（只判定、不改值）
                        ModelConfig ok;
                        ok.batchSize = in->batchRange().max;
                        bool passed = true;
                        try { applyInputShape(eng, in->name, l, 3, ok); }
                        catch (...) { passed = false; }
                        check(passed && ok.batchSize == in->batchRange().max,
                              "[13c] applyInputShape: batch == max -> ok, echo-back unchanged");
                    }
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