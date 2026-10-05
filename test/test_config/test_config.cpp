// =============================================================================
//  test/test_config/test_config.cpp
// -----------------------------------------------------------------------------
//  config 模块测试：
//    [1] loadModelConfig 读完整 INI（含所有必填 + 可选）
//    [2] 缺必填字段抛异常（每个必填字段各测一次）
//    [3] loadClassNamesFile 读正常 TXT
//    [4] TXT 格式错误抛异常（少字段 / RGB 越界）
//    [5] 空 TXT / 只有注释 → 返回空 vector
//    [6] input.layout：缺省为空 / 合法解析（含 5D） / 非法报错
//
//  临时文件用绝对路径（Paths::resolve 对绝对路径原样返回，不走工程根）
// =============================================================================
#include "trt_alpha/core/config.hpp"
#include "trt_alpha/core/model_config.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using trt_alpha::core::loadClassNamesFile;
using trt_alpha::core::loadModelConfig;
using trt_alpha::core::ModelConfig;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

//! 写临时文件，返回绝对路径。
fs::path writeTempFile(const std::string& name, const std::string& content)
{
    const fs::path p = fs::temp_directory_path() / name;
    std::ofstream out(p, std::ios::binary);
    out << content;
    return p;
}

//! 删临时文件（忽略错误）。
void removeTempFile(const fs::path& p)
{
    std::error_code ec;
    fs::remove(p, ec);
}

}  // namespace

int main()
{
    std::cout << "=== config tests ===\n";

        // [1] 完整 INI
    {
        const std::string ini =
            "# test config\n"
            "[model]\n"
            "engine = data/yolov8/yolov8n.trt\n"
            "num_class = 80\n"
            "class_names_file = data/classes/coco80.txt\n"
            "input_output_names = images,output0\n"
            "\n"
            "[input]\n"
            "layout = nhwc\n"
            "dst_h = 640\n"
            "dst_w = 640\n"
            "batch_size = 8\n"
            "max_batch_size = 4\n"
            "\n"
            "[postprocess]\n"
            "conf_thresh = 0.25\n"
            "iou_thresh = 0.45\n"
            "top_k = 300\n"
            "\n"
            "[output]\n"
            "save_path = data/output\n";

        const fs::path p = writeTempFile("test_yolov8.ini", ini);
        try
        {
            ModelConfig cfg = loadModelConfig(p.string());

            // ---- 通用字段 ----
            check(cfg.engine == "data/yolov8/yolov8n.trt",  "[1] engine");
            check(cfg.classNamesFile == "data/classes/coco80.txt",
                  "[1] class_names_file");
            check(cfg.batchSize == 8,                       "[1] batch_size");
            check(cfg.maxBatchSize == 4,                    "[1] max_batch_size");
            check(cfg.dstH == 640 && cfg.dstW == 640,      "[1] dst_h/w");
            check(cfg.layout == trt_alpha::core::Layout::NHWC, "[1] input.layout");
            check(cfg.inputOutputNames.size() == 2,         "[1] io size");
            check(cfg.inputOutputNames[0] == "images",      "[1] io[0]");
            check(cfg.inputOutputNames[1] == "output0",     "[1] io[1]");

            // ---- 模型特有字段：从 extras 读（两种 key 都能命中）----
            check(cfg.getInt("num_class", 0) == 80,         "[1] extras num_class (short)");
            check(cfg.getInt("model.num_class", 0) == 80,   "[1] extras num_class (full)");
            check(cfg.getFloat("conf_thresh", 0.f) == 0.25f, "[1] extras conf_thresh");
            check(cfg.getFloat("iou_thresh", 0.f) == 0.45f,  "[1] extras iou_thresh");
            check(cfg.getInt("top_k", 0) == 300,            "[1] extras top_k");
            check(cfg.getString("save_path", "") == "data/output",
                  "[1] extras save_path");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [1] threw: " << e.what() << "\n";
            ++g_failures;
        }
        removeTempFile(p);
    }

    // ---------------------------------------------------------------
    // [2] 缺必填字段
    // ---------------------------------------------------------------
    {
        // 2a：缺 engine
        {
            const std::string ini =
                "[model]\n"
                "num_class = 80\n"
                "class_names_file = x.txt\n"
                "input_output_names = a,b\n";
            const fs::path p = writeTempFile("test_2a.ini", ini);
            bool threw = false;
            try { (void)loadModelConfig(p.string()); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[2a] missing engine throws");
            removeTempFile(p);
        }
        // 2b：缺 num_class —— 现在不抛（模型自己校验）
        {
            const std::string ini =
                "[model]\n"
                "engine = x.trt\n"
                "class_names_file = x.txt\n"
                "input_output_names = a,b\n";
            const fs::path p = writeTempFile("test_2b.ini", ini);
            bool ok = false;
            try
            {
                ModelConfig cfg = loadModelConfig(p.string());
                // 通用字段都读到了；num_class 缺失 -> getInt 返回 fallback
                ok = (cfg.getInt("num_class", -1) == -1);
            }
            catch (...) { ok = false; }
            check(ok, "[2b] missing num_class does NOT throw; getInt returns fallback");
            removeTempFile(p);
        }
        // 2c：缺 class_names_file
        {
            const std::string ini =
                "[model]\n"
                "engine = x.trt\n"
                "num_class = 80\n"
                "input_output_names = a,b\n";
            const fs::path p = writeTempFile("test_2c.ini", ini);
            bool threw = false;
            try { (void)loadModelConfig(p.string()); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[2c] missing class_names_file throws");
            removeTempFile(p);
        }
        // 2d：缺 input_output_names
        {
            const std::string ini =
                "[model]\n"
                "engine = x.trt\n"
                "num_class = 80\n"
                "class_names_file = x.txt\n";
            const fs::path p = writeTempFile("test_2d.ini", ini);
            bool threw = false;
            try { (void)loadModelConfig(p.string()); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[2d] missing input_output_names throws");
            removeTempFile(p);
        }
                // 2e：num_class = 0 —— 现在不抛；模型 init() 里自己校验
        {
            const std::string ini =
                "[model]\n"
                "engine = x.trt\n"
                "num_class = 0\n"
                "class_names_file = x.txt\n"
                "input_output_names = a,b\n";
            const fs::path p = writeTempFile("test_2e.ini", ini);
            bool ok = false;
            try
            {
                ModelConfig cfg = loadModelConfig(p.string());
                ok = (cfg.getInt("num_class", -1) == 0);
            }
            catch (...) { ok = false; }
            check(ok, "[2e] num_class == 0 does NOT throw (validated in model init)");
            removeTempFile(p);
        }
    }

    // ---------------------------------------------------------------
    // [3] loadClassNamesFile 正常
    // ---------------------------------------------------------------
    {
        const std::string txt =
            "# COCO subset\n"
            "person 24 100 255\n"
            "bicycle 70 195 152\n"
            "\n"
            "car 207 92 231\n"
            "  # indented comment\n"
            "  truck 50 50 50  \n";   // 前后有空白

        const fs::path p = writeTempFile("test_classes.txt", txt);
        try
        {
            auto classes = loadClassNamesFile(p.string());
            check(classes.size() == 4, "[3] 4 classes");
            check(classes[0].name == "person", "[3] class 0 name");
            check(classes[0].r == 24 && classes[0].g == 100 && classes[0].b == 255,
                  "[3] class 0 color RGB");
            check(classes[2].name == "car", "[3] class 2 name");
            check(classes[3].name == "truck", "[3] class 3 name (trimmed)");
            check(classes[3].r == 50, "[3] class 3 color");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [3] threw: " << e.what() << "\n";
            ++g_failures;
        }
        removeTempFile(p);
    }

    // ---------------------------------------------------------------
    // [4] TXT 格式错误
    // ---------------------------------------------------------------
    {
        // 4a：少字段
        {
            const std::string txt = "person 24 100\n";   // 只有 2 个数字
            const fs::path p = writeTempFile("test_4a.txt", txt);
            bool threw = false;
            try { (void)loadClassNamesFile(p.string()); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[4a] malformed line throws");
            removeTempFile(p);
        }
        // 4b：RGB 越界
        {
            const std::string txt = "person 300 100 255\n";
            const fs::path p = writeTempFile("test_4b.txt", txt);
            bool threw = false;
            try { (void)loadClassNamesFile(p.string()); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[4b] RGB out of range throws");
            removeTempFile(p);
        }
        // 4c：文件不存在
        {
            const fs::path p = fs::temp_directory_path() / "no_such_classes_12345.txt";
            bool threw = false;
            try { (void)loadClassNamesFile(p.string()); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "[4c] nonexistent file throws");
        }
    }

    // ---------------------------------------------------------------
    // [5] 空 TXT / 只有注释
    // ---------------------------------------------------------------
    {
        const std::string txt = "# only comments\n\n; also comment\n";
        const fs::path p = writeTempFile("test_empty.txt", txt);
        try
        {
            auto classes = loadClassNamesFile(p.string());
            check(classes.empty(), "[5] empty result for comment-only file");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [5] threw: " << e.what() << "\n";
            ++g_failures;
        }
        removeTempFile(p);
    }

    // ---------------------------------------------------------------
    // [6] input.layout：缺省为空 / 合法解析 / 非法报错
    // ---------------------------------------------------------------
    {
        const std::string head =
            "[model]\n"
            "engine = a.trt\n"
            "class_names_file = data/classes/coco80.txt\n"
            "input_output_names = images,output0\n"
            "\n"
            "[input]\n";

        // 缺省：不写 layout → 空布局（由模型规范布局兜底）
        const fs::path p0 = writeTempFile("test_layout_none.ini", head);
        try
        {
            const ModelConfig cfg = loadModelConfig(p0.string());
            check(cfg.layout.empty(), "[6a] no input.layout -> empty (model default)");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [6a] threw: " << e.what() << "\n";
            ++g_failures;
        }
        removeTempFile(p0);

        // 合法（大小写不敏感、支持 5D）
        const fs::path p1 = writeTempFile("test_layout_ok.ini", head + "layout = NCDHW\n");
        try
        {
            const ModelConfig cfg = loadModelConfig(p1.string());
            check(cfg.layout == trt_alpha::core::Layout::NCDHW,
                  "[6b] input.layout = NCDHW parsed");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [6b] threw: " << e.what() << "\n";
            ++g_failures;
        }
        removeTempFile(p1);

        // 非法：未知字母必须报错，绝不静默忽略
        const fs::path p2 = writeTempFile("test_layout_bad.ini", head + "layout = nchq\n");
        bool threw = false;
        try { (void)loadModelConfig(p2.string()); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[6c] invalid input.layout -> throws");
        removeTempFile(p2);
    }

    std::cout << "====================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}