// =============================================================================
//  trt_alpha :: core :: logger
// -----------------------------------------------------------------------------
//  应用日志设施 + TensorRT ILogger 桥接。
//
//  【日志级别】
//    DEBUG / INFO / WARN / ERROR
//    编译期通过 TRT_ALPHA_LOG_MIN_LEVEL 控制（CMake 按构建类型自动设）：
//      * Debug  构建 → DEBUG（全开，冗余日志，崩溃可诊断）
//      * Release 构建 → INFO（只保留关键信息，DEBUG 零开销）
//
//  【日志格式】
//    [2026-09-25 15:30:12.345] [DEBUG] [tid=12345] [yolov8.cpp:314 postprocess] message
//    包含：时间戳(ms) / 级别 / 线程 ID / 文件:行号 / 函数名 / 消息
//    这样"崩溃时看日志能迅速定位"。
//
//  【多线程安全】
//    * 线程本地 ostringstream 拼接（避免竞争）
//    * 全局锁保证"整行原子输出"（不撕裂）
//    * 与 TrtLoggerAdapter 共用同一把锁
//
//  【崩溃捕获】（可选，在 main / test 里调用）
//    installCrashHandler() —— 捕获 SIGSEGV / SIGABRT / 未处理异常，
//    打 ERROR 日志后退出，Release 下也能"留下遗言"。
//
//  【不做】
//    * 日志轮转、网络日志、结构化日志（YAGNI）
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/data_type.hpp"

#include <NvInfer.h>

#include <iostream>
#include <mutex>
#include <sstream>

// -----------------------------------------------------------------------------
//  级别常量（数值越大越严重）
// -----------------------------------------------------------------------------
#define TRT_ALPHA_LOG_LEVEL_DEBUG 0
#define TRT_ALPHA_LOG_LEVEL_INFO  1
#define TRT_ALPHA_LOG_LEVEL_WARN  2
#define TRT_ALPHA_LOG_LEVEL_ERROR 3

#ifndef TRT_ALPHA_LOG_MIN_LEVEL
#    define TRT_ALPHA_LOG_MIN_LEVEL TRT_ALPHA_LOG_LEVEL_DEBUG
#endif

namespace trt_alpha::core::detail {

//! 全局日志锁：保证"整行输出"原子。
inline std::mutex& logMutex() noexcept
{
    static std::mutex m;
    return m;
}

//! 线程本地输出流：避免拼接阶段的数据竞争。
//! 每次调用清空并复用，避免频繁分配。
inline std::ostringstream& tlsStream()
{
    thread_local std::ostringstream oss;
    oss.str("");
    oss.clear();
    return oss;
}

//! 生成日志前缀："[时间戳] [级别] [tid] [文件:行号 函数名] "
//! 由 logger.cpp 实现。
std::string logPrefix(const char* level,
                      const char* file,
                      int line,
                      const char* func);

}  // namespace trt_alpha::core::detail

// -----------------------------------------------------------------------------
//  应用日志宏
//
//  注意：__VA_ARGS__ 外面【不加括号】。
//  原因：__VA_ARGS__ 展开后形如 `"..." << msg`，它是一个"流插入表达式"。
//  如果加括号变成 `("..." << msg)`，左操作数就成了字符串字面量，
//  而 C++ 没有 `const char* << T` 这个重载，会报 C2296 / C2297。
//
//  如果调用点含逗号（例如 `duration<double, std::milli>`），
//  预处理器会把逗号当参数分隔符，报 C4002。
//  解法：在【调用点】给含逗号的子表达式额外加一层圆括号：
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
//  TensorRT ILogger 桥接
// -----------------------------------------------------------------------------
namespace trt_alpha::core {

//! 把 TensorRT 内部日志转发到应用日志。
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

//! 全局 TRT logger 单例（builder / runtime 共用）。
inline TrtLoggerAdapter& trtLogger() noexcept
{
    static TrtLoggerAdapter instance;
    return instance;
}

// -----------------------------------------------------------------------------
//  崩溃捕获（可选，在 main / test 里调用）
// -----------------------------------------------------------------------------
//! 安装崩溃处理器：
//!   * Linux: SIGSEGV / SIGABRT / SIGFPE / SIGILL
//!   * Windows: SetUnhandledExceptionFilter
//! 崩溃时打 ERROR 日志并 flush，然后退出。
//! 幂等（重复调用只生效一次）。
void installCrashHandler() noexcept;

}  // namespace trt_alpha::core

// -----------------------------------------------------------------------------
//  内存分配框图 log
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

//! 打一个多行框图（原子输出，不撕裂）。
void logAllocBox(const AllocInfo& info);

}  // namespace trt_alpha::core::detail