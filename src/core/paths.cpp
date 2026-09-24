// =============================================================================
//  trt_alpha :: core :: paths（实现）
// =============================================================================
#include "trt_alpha/core/paths.hpp"
#include "trt_alpha/core/logger.hpp"   //  新增

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace trt_alpha::core {
namespace {

// ---- 探测结果缓存（首次 root() 时写入，之后只读；读写均在同一把锁下）----
std::mutex g_mutex;
std::string g_override;   //!< --root 传入的目录
bool g_ready = false;
fs::path g_root;
std::string g_source;

bool isAscii(const std::string& text)
{
    for (const unsigned char c : text)
    {
        if (c > 127)
        {
            return false;
        }
    }
    return true;
}

//! "根标志"判定：configs/ 子目录（仓库/发布包都有）或 .trt_alpha_root 手工标记。
bool looksLikeRoot(const fs::path& dir)
{
    std::error_code ec;
    if (fs::is_directory(dir / "configs", ec))
    {
        return true;
    }
    return fs::exists(dir / ".trt_alpha_root", ec);
}

//! 从 dir 起逐级向上找根标志；找不到返回空路径。
fs::path searchUpward(const fs::path& dir)
{
    if (dir.empty())
    {
        return {};
    }
    std::error_code ec;
    fs::path current = fs::absolute(dir, ec);
    if (ec)
    {
        current = dir;
    }
    for (;;)
    {
        if (looksLikeRoot(current))
        {
            return current;
        }
        const fs::path parent = current.parent_path();
        if (parent.empty() || parent == current)
        {
            return {};
        }
        current = parent;
    }
}

//! 可执行文件所在目录（Windows: GetModuleFileNameW / Linux: /proc/self/exe）。
fs::path executableDir()
{
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;)
    {
        const DWORD length =
            ::GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0)
        {
            return {};
        }
        if (length < buffer.size())
        {
            buffer.resize(length);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
    return fs::path(buffer).parent_path();
#else
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (ec || exe.empty())
    {
        return {};
    }
    return exe.parent_path();
#endif
}

//! 按优先级探测根目录；source 输出命中的规则名。
fs::path detect(std::string& source)
{
    std::error_code ec;

    // ---- 1. --root 显式指定 ----
    if (!g_override.empty())
    {
        const fs::path path = fs::absolute(fs::path(g_override), ec);
        if (!fs::is_directory(path, ec))
        {
            throw std::runtime_error("--root is not a directory: " + g_override);
        }
        source = "command line (--root)";
        return path;
    }

    // ---- 2. 环境变量 ----
    if (const char* env = std::getenv("TRT_ALPHA_ROOT"); env != nullptr && *env != '\0')
    {
        const fs::path path = fs::absolute(fs::path(env), ec);
        if (!fs::is_directory(path, ec))
        {
            throw std::runtime_error("TRT_ALPHA_ROOT is not a directory: " +
                                     std::string(env));
        }
        source = "environment TRT_ALPHA_ROOT";
        return path;
    }

    // ---- 3. 可执行文件向上找 ----
    if (const fs::path found = searchUpward(executableDir()); !found.empty())
    {
        source = "executable location";
        return found;
    }

    // ---- 4. 编译期写入的源码根 ----
#ifdef TRT_ALPHA_ROOT_DIR
    {
        fs::path baked(TRT_ALPHA_ROOT_DIR);
        baked.make_preferred();
        if (fs::is_directory(baked, ec))
        {
            source = "compile-time TRT_ALPHA_ROOT_DIR";
            return baked;
        }
    }
#endif

    // ---- 5. 当前工作目录向上找 ----
    if (const fs::path found = searchUpward(fs::current_path(ec)); !found.empty())
    {
        source = "working directory";
        return found;
    }

    // ---- 6. 兜底：工作目录 ----
    source = "working directory (no root marker found)";
    return fs::current_path(ec);
}

}  // namespace

void Paths::setOverride(const std::string& dir)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ready)
    {
        TRT_LOG_ERROR("Paths::setOverride called after root() was already resolved");
        throw std::logic_error("Paths::setOverride() must be called before the first root()");
    }
    g_override = dir;
    TRT_LOG_INFO("Paths: override set to '" << dir << "'");
}

const fs::path& Paths::root()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ready)
    {
        fs::path detected = detect(g_source);
        g_root = std::move(detected);
        g_ready = true;

        // 打一次即可（首次探测时）
        TRT_LOG_INFO("Paths: root = " << g_root.string()
                     << "  (source: " << g_source << ")");
    }
    return g_root;
}

std::string Paths::rootSource()
{
    (void)root();
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_source;
}

fs::path Paths::toPath(const std::string& text)
{
    fs::path path(text);
    if (isAscii(text))
    {
        return path;
    }
#ifdef _WIN32
    std::error_code ec;
    if (!fs::exists(path, ec))
    {
        const fs::path utf8 = fs::u8path(text);
        if (fs::exists(utf8, ec))
        {
            return utf8;
        }
    }
#endif
    return path;
}

fs::path Paths::resolve(const std::string& text)
{
    if (text.empty())
    {
        return {};
    }
    const fs::path path = toPath(text);
    if (path.is_absolute())
    {
        return path;
    }
    fs::path joined = root() / path;
    joined.make_preferred();
    return joined;
}

fs::path Paths::requireFile(const std::string& text, const std::string& role)
{
    const fs::path path = resolve(text);
    std::error_code ec;
    if (!fs::exists(path, ec))
    {
        TRT_LOG_ERROR("Paths: " << role << " not found: " << toDisplay(path));
        throw std::runtime_error(role + " not found: " + toDisplay(path) +
                                 "\n  project root : " + toDisplay(root()) +
                                 "  (detected from " + rootSource() + ")" +
                                 "\n  hint         : pass an absolute path, or set "
                                 "TRT_ALPHA_ROOT=<dir> / --root <dir>");
    }
    return path;
}

std::string Paths::toDisplay(const fs::path& path)
{
    return path.string();
}

}  // namespace trt_alpha::core