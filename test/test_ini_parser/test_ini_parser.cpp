// =============================================================================
//  test/test_ini_parser/test_ini_parser.cpp
// -----------------------------------------------------------------------------
//  IniParser 测试（用 parseString，不碰文件系统）：
//    [1] 扁平 key（无节）
//    [2] 节内 key
//    [3] 注释 / 空行 / 空白
//    [4] 类型读取（int / float / list）
//    [5] 语法错误抛异常
// =============================================================================
#include "trt_alpha/core/ini_parser.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using trt_alpha::core::IniParser;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

}  // namespace

int main()
{
    std::cout << "=== IniParser tests ===\n";

    // [1] 无节
    {
        IniParser ini = IniParser::parseString(
            "batch_size = 8\n"
            "conf_thresh = 0.25\n");

        check(ini.has("batch_size"),           "[1] has batch_size");
        check(ini.getInt("batch_size", 0) == 8, "[1] batch_size == 8");
        check(ini.getFloat("conf_thresh", 0.f) == 0.25f, "[1] conf_thresh == 0.25");
    }

    // [2] 有节
    {
        IniParser ini = IniParser::parseString(
            "[model]\n"
            "engine = data/yolov8/yolov8n.trt\n"
            "num_class = 80\n"
            "\n"
            "[input]\n"
            "dst_h = 640\n"
            "dst_w = 640\n");

        check(ini.has("engine", "model"),           "[2] has model.engine");
        check(ini.has("model.engine"),              "[2] has model.engine (dot form)");
        check(ini.getString("engine", "model", "") == "data/yolov8/yolov8n.trt",
              "[2] engine value");
        check(ini.getInt("num_class", "model", 0) == 80, "[2] num_class");
        check(ini.getInt("dst_h", "input", 0) == 640, "[2] dst_h");
        check(ini.getInt("dst_w", "input", 0) == 640, "[2] dst_w");
    }

    // [3] 注释 / 空行 / 空白
    {
        IniParser ini = IniParser::parseString(
            "# comment line\n"
            "; another comment\n"
            "\n"
            "  key1  =  value1  \n"
            "key2 = value2 # inline comment\n"
            "\n");

        check(ini.getString("key1") == "value1", "[3] key1 trimmed");
        check(ini.getString("key2") == "value2", "[3] inline comment stripped");
    }

    // [4] 类型 + list
    {
        IniParser ini = IniParser::parseString(
            "n = 42\n"
            "f = 3.14\n"
            "names = images,output0,output1\n"
            "names_sp = a b c\n");

        check(ini.getInt("n", 0) == 42, "[4] int");
        check(ini.getFloat("f", 0.f) == 3.14f, "[4] float");
        auto lst = ini.getStringList("names");
        check(lst.size() == 3, "[4] list size 3");
        check(lst[0] == "images" && lst[1] == "output0" && lst[2] == "output1",
              "[4] list contents");
        auto lst2 = ini.getStringList("names_sp", "", ' ');
        check(lst2.size() == 3, "[4] space-separated list");
    }

    // [5] 语法错误
    {
        bool threw = false;
        try { (void)IniParser::parseString("[unclosed\nkey=1\n"); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[5] unclosed section header throws");

        threw = false;
        try { (void)IniParser::parseString("no_equal_sign_line\n"); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[5] missing '=' throws");

        threw = false;
        try
        {
            auto ini = IniParser::parseString("key = notanint\n");
            (void)ini.getInt("key", 0);
        }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "[5] bad int throws");
    }

    std::cout << "=======================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}