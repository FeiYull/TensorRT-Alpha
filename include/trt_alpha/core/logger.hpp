// =============================================================================
//  trt_alpha :: core :: logger
// -----------------------------------------------------------------------------
//  Application logging facility + TensorRT ILogger bridge.
//
//  [Log levels]
//    DEBUG / INFO / WARN / ERROR
//    Controlled at compile time via TRT_ALPHA_LOG_MIN_LEVEL
//    (CMake sets it automatically per build type):
//      * Debug   build -> DEBUG (all on, verbose logs, crash-diagnosable)
//      * Release build -> INFO (key info only, DEBUG is zero-cost)
//
//  [Log format]
//    [2026-09-25 15:30:12.345] [DEBUG] [tid=12345] [yolov8.cpp:314 postprocess] message
//    Contains: timestamp(ms) / level / thread ID / file:line / function name / message
//    so that "when it crashes, the log pinpoints the location fast".
//
//  [Thread safety]
//    * thread-local ostringstream for assembling (avoids races)
//    * a global lock guarantees "whole-line atomic output" (no tearing)
//    * shares the same lock with TrtLoggerAdapter
//
//  [Crash capture] (optional; call from main / test)
//    installCrashHandler() -- catches SIGSEGV / SIGABRT / unhandled exceptions,
//    logs at ERROR then exits, leaving a "last word" even in Release.
//
//  [Not doing]
//    * log rotation, network logging, structured logging (YAGNI)
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"

#include <NvInfer.h>

#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
//  Level constants (larger value = more severe)
// -----------------------------------------------------------------------------
#define TRT_ALPHA_LOG_LEVEL_DEBUG 0
#define TRT_ALPHA_LOG_LEVEL_INFO  1
#define TRT_ALPHA_LOG_LEVEL_WARN  2
#define TRT_ALPHA_LOG_LEVEL_ERROR 3

#ifndef TRT_ALPHA_LOG_MIN_LEVEL
#    define TRT_ALPHA_LOG_MIN_LEVEL TRT_ALPHA_LOG_LEVEL_DEBUG
#endif

namespace trt_alpha::core::detail {

//! Global log lock: guarantees atomic whole-line output.
inline std::mutex& logMutex() noexcept
{
    static std::mutex m;
    return m;
}

//! Thread-local output stream: avoids data races during assembly.
//! Cleared and reused on every call to avoid frequent allocations.
inline std::ostringstream& tlsStream()
{
    thread_local std::ostringstream oss;
    oss.str("");
    oss.clear();
    return oss;
}

//! Build the log prefix: "[timestamp] [level] [tid] [file:line function] "
//! Implemented in logger.cpp.
std::string logPrefix(const char* level,
                      const char* file,
                      int line,
                      const char* func);

}  // namespace trt_alpha::core::detail

// -----------------------------------------------------------------------------
//  Application log macros
//
//  Note: do NOT wrap __VA_ARGS__ in parentheses.
//  Reason: after expansion __VA_ARGS__ looks like `"..." << msg`, i.e. a
//  "stream-insertion expression". Adding parentheses turns the left operand
//  into a string literal, and C++ has no `const char* << T` overload,
//  which triggers C2296 / C2297.
//
//  If the call site contains a comma (e.g. `duration<double, std::milli>`),
//  the preprocessor treats the comma as an argument separator and reports C4002.
//  Fix: add an extra pair of parentheses around the comma-containing subexpression
//  at the [call site]:
//      TRT_LOG_DEBUG("... " << (std::chrono::duration<double, std::milli>(a-b).count())
//                   << " ms");
// -----------------------------------------------------------------------------
#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_DEBUG
#    define TRT_LOG_DEBUG(...)                                                    \
        do {                                                                      \
            auto& _oss = ::trt_alpha::core::detail::tlsStream();                  \
            _oss << ::trt_alpha::core::detail::logPrefix("DEBUG", __FILE__,       \
                        __LINE__, __func__)                                       \
                 << __VA_ARGS__ << '\n';                                          \
            std::lock_guard<std::mutex> _lk(::trt_alpha::core::detail::logMutex());\
            std::cout << _oss.str();                                              \
        } while (0)
#else
#    define TRT_LOG_DEBUG(...) do { } while (0)
#endif

#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_INFO
#    define TRT_LOG_INFO(...)                                                     \
        do {                                                                      \
            auto& _oss = ::trt_alpha::core::detail::tlsStream();                  \
            _oss << ::trt_alpha::core::detail::logPrefix("INFO ", __FILE__,       \
                        __LINE__, __func__)                                       \
                 << __VA_ARGS__ << '\n';                                          \
            std::lock_guard<std::mutex> _lk(::trt_alpha::core::detail::logMutex());\
            std::cout << _oss.str();                                              \
        } while (0)
#else
#    define TRT_LOG_INFO(...) do { } while (0)
#endif

#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_WARN
#    define TRT_LOG_WARN(...)                                                     \
        do {                                                                      \
            auto& _oss = ::trt_alpha::core::detail::tlsStream();                  \
            _oss << ::trt_alpha::core::detail::logPrefix("WARN ", __FILE__,       \
                        __LINE__, __func__)                                       \
                 << __VA_ARGS__ << '\n';                                          \
            std::lock_guard<std::mutex> _lk(::trt_alpha::core::detail::logMutex());\
            std::cerr << _oss.str();                                              \
        } while (0)
#else
#    define TRT_LOG_WARN(...) do { } while (0)
#endif

#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_ERROR
#    define TRT_LOG_ERROR(...)                                                    \
        do {                                                                      \
            auto& _oss = ::trt_alpha::core::detail::tlsStream();                  \
            _oss << ::trt_alpha::core::detail::logPrefix("ERROR", __FILE__,       \
                        __LINE__, __func__)                                       \
                 << __VA_ARGS__ << '\n';                                          \
            std::lock_guard<std::mutex> _lk(::trt_alpha::core::detail::logMutex());\
            std::cerr << _oss.str();                                              \
        } while (0)
#else
#    define TRT_LOG_ERROR(...) do { } while (0)
#endif

// -----------------------------------------------------------------------------
//  TensorRT ILogger bridge
// -----------------------------------------------------------------------------
namespace trt_alpha::core {

//! Forward TensorRT internal logs to the application log.
class TrtLoggerAdapter final : public nvinfer1::ILogger
{
public:
    explicit TrtLoggerAdapter(Severity minSeverity = Severity::kINFO) noexcept
        : m_minSeverity(minSeverity)
    {
    }

    nvinfer1::ILogger& trtLogger() noexcept { return *this; }

    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity > m_minSeverity)
        {
            return;
        }

        const char* tag = "INFO ";
        bool toStderr = false;
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: tag = "ERROR"; toStderr = true;  break;
        case Severity::kERROR:          tag = "ERROR"; toStderr = true;  break;
        case Severity::kWARNING:        tag = "WARN "; toStderr = true;  break;
        case Severity::kINFO:           tag = "INFO "; toStderr = false; break;
        case Severity::kVERBOSE:        tag = "DEBUG"; toStderr = false; break;
        }

        auto& oss = detail::tlsStream();
        oss << "[TRT " << tag << "] " << msg << '\n';

        std::lock_guard<std::mutex> lk(detail::logMutex());
        (toStderr ? std::cerr : std::cout) << oss.str();
    }

    void setMinSeverity(Severity s) noexcept { m_minSeverity = s; }

private:
    Severity m_minSeverity;
};

//! Global TRT logger singleton (shared by builder / runtime).
inline TrtLoggerAdapter& trtLogger() noexcept
{
    static TrtLoggerAdapter instance;
    return instance;
}

// -----------------------------------------------------------------------------
//  Crash capture (optional; call from main / test)
// -----------------------------------------------------------------------------
//! Install the crash handler:
//!   * Linux: SIGSEGV / SIGABRT / SIGFPE / SIGILL
//!   * Windows: SetUnhandledExceptionFilter
//! Logs at ERROR and flushes before exiting on a crash.
//! Idempotent (repeated calls take effect only once).
void installCrashHandler() noexcept;

}  // namespace trt_alpha::core

// -----------------------------------------------------------------------------
//  Memory allocation box log
// -----------------------------------------------------------------------------
namespace trt_alpha::core::detail {

struct AllocInfo
{
    const char* name = "";
    int batch = 0;
    int channels = 0;
    int height = 0;
    int width = 0;
    DataType dtype = DataType::UInt8;
    std::size_t bytes = 0;
    MemorySpace space = MemorySpace::Host;
};

//! Print a multi-line box (atomic output, no tearing).
void logAllocBox(const AllocInfo& info);

//! Render several "content lines" into a text box and atomically output it to stdout.
//!   * top/bottom border uses hLine (default '=', i.e. the "double horizontal line" style),
//!     left/right border '|'
//!   * width adapts to the longest line (content is not truncated)
//!   * output is [without] a log prefix, and is emitted in Release builds too (INFO semantics)
//! Used for display blocks like [CONFIG] that should be seen on one screen.
void logBox(const std::vector<std::string>& lines, char hLine = '=');

}  // namespace trt_alpha::core::detail
