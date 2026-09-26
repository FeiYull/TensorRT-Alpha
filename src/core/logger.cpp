// =============================================================================
//  trt_alpha :: core :: logger（实现）
// =============================================================================
#include "trt_alpha/core/logger.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <csignal>
#    include <execinfo.h>
#endif

namespace trt_alpha::core::detail {
namespace {

//! 从 __FILE__ 提取 basename（"src/core/logger.cpp" → "logger.cpp"）。
const char* basenameOf(const char* path) noexcept
{
    const char* p = std::strrchr(path, '/');
    if (p == nullptr) { p = std::strrchr(path, '\\'); }
    return (p == nullptr) ? path : (p + 1);
}

//! 生成时间戳字符串 "[2026-09-25 15:30:12.345]"。
std::string timestampString()
{
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto t = clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) % 1000;

    std::tm tmBuf{};
#ifdef _WIN32
    localtime_s(&tmBuf, &t);
#else
    localtime_r(&t, &tmBuf);
#endif

    char buf[64];
    std::snprintf(buf, sizeof(buf),
                  "[%04d-%02d-%02d %02d:%02d:%02d.%03d]",
                  tmBuf.tm_year + 1900, tmBuf.tm_mon + 1, tmBuf.tm_mday,
                  tmBuf.tm_hour, tmBuf.tm_min, tmBuf.tm_sec,
                  static_cast<int>(ms.count()));
    return buf;
}

//! 线程 ID 字符串 "[tid=12345]"。
std::string threadIdString()
{
    std::ostringstream oss;
    oss << "[tid=" << std::this_thread::get_id() << "]";
    return oss.str();
}

}  // namespace

std::string logPrefix(const char* level,
                      const char* file,
                      int line,
                      const char* func)
{
    std::ostringstream oss;
    oss << timestampString() << ' '
        << '[' << level << "] "
        << threadIdString() << ' '
        << '[' << basenameOf(file) << ':' << line << ' ' << func << "] ";
    return oss.str();
}

// =============================================================================
//  内存分配框图 log
// =============================================================================
namespace {

constexpr int kBoxWidth = 68;

void appendLine(std::ostringstream& oss, const std::string& content)
{
    const int innerWidth = kBoxWidth - 2;
    oss << "| " << content;
    if (static_cast<int>(content.size()) + 2 < innerWidth)
    {
        oss << std::string(innerWidth - 2 - content.size(), ' ');
    }
    oss << " |\n";
}

std::string formatDouble(double v, int precision)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

}  // namespace

void logAllocBox(const AllocInfo& info)
{
#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_DEBUG
    const std::size_t elemSize = sizeOf(info.dtype);
    const double mb = static_cast<double>(info.bytes) / (1024.0 * 1024.0);

    std::ostringstream oss;
    oss << "+" << std::string(kBoxWidth - 2, '-') << "+\n";

    { std::ostringstream l; l << "[ALLOC] " << (info.name ? info.name : ""); appendLine(oss, l.str()); }
    { std::ostringstream l; l << "  batch    = " << info.batch;   appendLine(oss, l.str()); }
    { std::ostringstream l; l << "  channels = " << info.channels; appendLine(oss, l.str()); }
    { std::ostringstream l; l << "  height   = " << info.height;   appendLine(oss, l.str()); }
    { std::ostringstream l; l << "  width    = " << info.width;    appendLine(oss, l.str()); }
    { std::ostringstream l; l << "  dtype    = " << nameOf(info.dtype)
                              << " (" << elemSize << " bytes)";    appendLine(oss, l.str()); }
    { std::ostringstream l; l << "  calc     = " << info.batch << " * " << info.channels
                              << " * " << info.height << " * " << info.width
                              << " * " << elemSize;                appendLine(oss, l.str()); }
    { std::ostringstream l; l << "  bytes    = " << info.bytes;    appendLine(oss, l.str()); }
    { std::ostringstream l; l << "  MB       = " << formatDouble(mb, 2) << " MB"; appendLine(oss, l.str()); }
    { std::ostringstream l; l << "  space    = "
                              << (info.space == MemorySpace::Device ? "Device" : "Host");
                                                                  appendLine(oss, l.str()); }

    oss << "+" << std::string(kBoxWidth - 2, '-') << "+\n";

    std::lock_guard<std::mutex> lk(logMutex());
    std::cout << oss.str();
#else
    (void)info;   // Release / MinSizeRel: 不编译，零开销
#endif
}

}  // namespace trt_alpha::core::detail

// =============================================================================
//  崩溃捕获
// =============================================================================
namespace trt_alpha::core {
namespace {

std::atomic<bool> g_crashHandlerInstalled{false};

//! 崩溃时打日志并 flush。
void onFatal(const char* what) noexcept
{
    // 直接用 fprintf（信号安全，不能走 iostream / mutex）
    std::fprintf(stderr,
                 "\n[FATAL] %s\n"
                 "[FATAL] crash handler caught a fatal error; "
                 "check the last DEBUG logs above for context.\n",
                 what);
    std::fflush(stderr);
}

#ifdef _WIN32

LONG WINAPI unhandledExceptionFilter(EXCEPTION_POINTERS* info)
{
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "unhandled exception code=0x%08lX",
                  static_cast<unsigned long>(info->ExceptionRecord->ExceptionCode));
    onFatal(buf);
    return EXCEPTION_EXECUTE_HANDLER;
}

void installImpl() noexcept
{
    ::SetUnhandledExceptionFilter(&unhandledExceptionFilter);
}

#else  // POSIX

void signalHandler(int sig) noexcept
{
    const char* name = "unknown";
    switch (sig)
    {
    case SIGSEGV: name = "SIGSEGV (segmentation fault)"; break;
    case SIGABRT: name = "SIGABRT (abort)";              break;
    case SIGFPE:  name = "SIGFPE (floating point)";      break;
    case SIGILL:  name = "SIGILL (illegal instruction)"; break;
    case SIGBUS:  name = "SIGBUS (bus error)";           break;
    }
    onFatal(name);

    // 还原默认 handler 并重新触发，保留 core dump 行为
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void installImpl() noexcept
{
    std::signal(SIGSEGV, &signalHandler);
    std::signal(SIGABRT, &signalHandler);
    std::signal(SIGFPE,  &signalHandler);
    std::signal(SIGILL,  &signalHandler);
    std::signal(SIGBUS,  &signalHandler);
}

#endif

}  // namespace

void installCrashHandler() noexcept
{
    bool expected = false;
    if (!g_crashHandlerInstalled.compare_exchange_strong(expected, true))
    {
        return;   // 已经装过
    }
    installImpl();
}

}  // namespace trt_alpha::core