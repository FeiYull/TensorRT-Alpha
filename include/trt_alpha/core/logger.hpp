// =============================================================================
//  trt_alpha :: core :: logger
// -----------------------------------------------------------------------------
//  应用日志设施 + TensorRT ILogger 桥接。
//
//  设计要点：
//    * 4 级：DEBUG / INFO / WARN / ERROR
//    * 输出：DEBUG/INFO → stdout；WARN/ERROR → stderr
//    * 多线程安全：线程本地 ostringstream 拼接 + 全局锁一次输出
//      （保证"整行原子"，不会撕裂成 [INFO ] [INFO ] ...）
//    * TRT 的 ILogger 由 TrtLoggerAdapter 桥接到同一套输出
//
//  不做：
//    * 日志轮转、网络日志、结构化日志（YAGNI）
//    * 时间戳、线程 ID（保持简洁；将来需要再加）
//
//  编译期级别过滤：
//    定义 TRT_ALPHA_LOG_MIN_LEVEL 可关闭低级别日志（见下方宏）。
//    默认不过滤（全输出）。
// =============================================================================
#pragma once

#include "trt_alpha/core/buffer_view.hpp"   // MemorySpace
#include "trt_alpha/core/data_type.hpp"     // DataType

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

}  // namespace trt_alpha::core::detail

// -----------------------------------------------------------------------------
//  应用日志宏
// -----------------------------------------------------------------------------
#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_DEBUG
#    define TRT_LOG_DEBUG(msg)                                                    \
        do {                                                                      \
            auto& _oss = ::trt_alpha::core::detail::tlsStream();                  \
            _oss << "[DEBUG] " << msg << '\n';                                    \
            std::lock_guard<std::mutex> _lk(::trt_alpha::core::detail::logMutex());\
            std::cout << _oss.str();                                              \
        } while (0)
#else
#    define TRT_LOG_DEBUG(msg) do { } while (0)
#endif

#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_INFO
#    define TRT_LOG_INFO(msg)                                                     \
        do {                                                                      \
            auto& _oss = ::trt_alpha::core::detail::tlsStream();                  \
            _oss << "[INFO ] " << msg << '\n';                                    \
            std::lock_guard<std::mutex> _lk(::trt_alpha::core::detail::logMutex());\
            std::cout << _oss.str();                                              \
        } while (0)
#else
#    define TRT_LOG_INFO(msg) do { } while (0)
#endif

#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_WARN
#    define TRT_LOG_WARN(msg)                                                     \
        do {                                                                      \
            auto& _oss = ::trt_alpha::core::detail::tlsStream();                  \
            _oss << "[WARN ] " << msg << '\n';                                    \
            std::lock_guard<std::mutex> _lk(::trt_alpha::core::detail::logMutex());\
            std::cerr << _oss.str();                                              \
        } while (0)
#else
#    define TRT_LOG_WARN(msg) do {  } while (0)
#endif

#if TRT_ALPHA_LOG_MIN_LEVEL <= TRT_ALPHA_LOG_LEVEL_ERROR
#    define TRT_LOG_ERROR(msg)                                                    \
        do {                                                                      \
            auto& _oss = ::trt_alpha::core::detail::tlsStream();                  \
            _oss << "[ERROR] " << msg << '\n';                                    \
            std::lock_guard<std::mutex> _lk(::trt_alpha::core::detail::logMutex());\
            std::cerr << _oss.str();                                              \
        } while (0)
#else
#    define TRT_LOG_ERROR(msg) do {  } while (0)
#endif

// -----------------------------------------------------------------------------
//  TensorRT ILogger 桥接
// -----------------------------------------------------------------------------
namespace trt_alpha::core {

//! 把 TensorRT 内部日志转发到应用日志。
//! 不直接用 TRT_LOG_* 宏（TRT 的 severity 参数是运行时值，宏是编译期）；
//! 内部直接走 detail::tlsStream + logMutex，与宏共享同一把锁，保证不撕裂。
class TrtLoggerAdapter final : public nvinfer1::ILogger
{
public:
    explicit TrtLoggerAdapter(Severity minSeverity = Severity::kINFO) noexcept
        : m_minSeverity(minSeverity)
    {
    }

    //! 给 createInferRuntime() / createInferBuilder() 用。
    nvinfer1::ILogger& trtLogger() noexcept { return *this; }

    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity > m_minSeverity)
        {
            return;
        }

        // 级别映射：TRT -> 应用
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

}  // namespace trt_alpha::core



// =============================================================================
//  内存分配框图 log
// -----------------------------------------------------------------------------
//  用途：模型 / 数据源在"分配关键缓冲"时调用，把
//    batch × 高 × 宽 × 通道 × 数据类型 → 字节数 → MB 用框图打出。
//
//  为什么在 logger 里：
//    * 它和日志共用同一把锁（不撕裂）
//    * 使用者只要 include logger.hpp 就能用
//    * 池内部只打 DEBUG（它不知道语义），框图由"知道语义的人"打
//
//  走 INFO 级别（Release 也打）。
// =============================================================================
namespace trt_alpha::core::detail {

struct AllocInfo
{
    const char* name = "";       //!< "input_nchw" / "output0" / "mask_proto"
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