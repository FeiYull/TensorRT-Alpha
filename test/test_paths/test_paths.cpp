// =============================================================================
//  test/test_paths/test_paths.cpp
// -----------------------------------------------------------------------------
//  Paths 测试：
//    [1] root() 返回目录 + rootSource() 非空
//    [2] resolve()：相对路径按 root 展开；绝对路径原样
//    [3] requireFile()：存在 OK；不存在抛异常
//    [4] setOverride 时序规则（必须在首次 root() 前）
//    [5] toPath / toDisplay 基本行为
//    [6] isUrl：URL 判定（scheme 形态），Windows 盘符不误判
// =============================================================================
#include "trt_alpha/core/paths.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using trt_alpha::core::Paths;

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
    std::cout << "=== Paths tests ===\n";

    // [1] root() / rootSource()
    {
        const fs::path& r = Paths::root();
        check(!r.empty(),                     "[1] root() not empty");
        check(fs::is_directory(r),            "[1] root() is a directory");
        check(!Paths::rootSource().empty(),   "[1] rootSource() not empty");
        std::cout << "       root = " << r << "\n";
        std::cout << "       source = " << Paths::rootSource() << "\n";
    }

    // [2] resolve()
    {
        const fs::path rel = Paths::resolve("configs/foo.ini");
        check(rel.is_absolute(),              "[2] relative -> absolute");
        check(rel.filename() == "foo.ini",    "[2] filename preserved");

        const fs::path abs("/some/abs/path");
        const fs::path r2 = Paths::resolve(abs.string());
        check(r2.is_absolute(),               "[2] absolute stays absolute");
    }

    // [3] requireFile()
    {
        // 不存在的文件抛异常
        bool threw = false;
        try
        {
            (void)Paths::requireFile("this_file_does_not_exist_12345.ini",
                                     "test config");
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        check(threw, "[3] requireFile throws for missing file");

        // 用临时文件验证 OK 路径（不依赖本机绝对路径）
        const fs::path tmp = fs::temp_directory_path() / "test_paths_ok.txt";
        {
            std::ofstream out(tmp);
            out << "ok";
        }
        try
        {
            const fs::path ok = Paths::requireFile(tmp.string(), "temp file");
            check(fs::exists(ok), "[3] requireFile OK for existing file");
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] [3] threw: " << e.what() << "\n";
            ++g_failures;
        }
        std::error_code ec;
        fs::remove(tmp, ec);
    }

    // [4] setOverride 时序规则
    //     注意：setOverride 必须在 root() 首次调用【之前】。
    //     本测试文件已经调过 root()，所以这里 setOverride 应该抛逻辑错误。
    {
        bool threw = false;
        try
        {
            Paths::setOverride("some/dir");
        }
        catch (const std::logic_error&)
        {
            threw = true;
        }
        check(threw, "[4] setOverride after root() throws logic_error");
    }

    // [5] toPath / toDisplay
    {
        const fs::path p = Paths::toPath("abc/def");
        check(p.string() == "abc/def" || p.string() == "abc\\def",
              "[5] toPath ASCII preserved");

        const std::string disp = Paths::toDisplay(fs::path("abc/def"));
        check(!disp.empty(), "[5] toDisplay non-empty");
    }

    // [6] isUrl：只认 <scheme>:// 形态；Windows 盘符 / 相对路径不误判
    {
        check( Paths::isUrl("rtsp://192.168.1.10:554/stream1"), "[6] rtsp:// is url");
        check( Paths::isUrl("RTSP://cam/live"),                  "[6] uppercase scheme is url");
        check( Paths::isUrl("https://example.com/a.mp4?t=1"),    "[6] https:// with query is url");
        check( Paths::isUrl("udp://239.0.0.1:1234"),             "[6] udp:// is url");
        check(!Paths::isUrl("data/bus.jpg"),                     "[6] relative path is not url");
        check(!Paths::isUrl("D:/data/bus.jpg"),                  "[6] windows drive is not url");
        check(!Paths::isUrl("D:\\data\\bus.jpg"),                "[6] windows backslash path is not url");
        check(!Paths::isUrl("C://foo"),                          "[6] 1-letter scheme rejected");
        check(!Paths::isUrl("http://"),                          "[6] empty host rejected");
        check(!Paths::isUrl("://host/x"),                        "[6] empty scheme rejected");
        check(!Paths::isUrl("data/bus.jpg://x"),                 "[6] scheme not at start rejected");
    }

    std::cout << "===================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}