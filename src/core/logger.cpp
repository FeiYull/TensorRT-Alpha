// =============================================================================
//  trt_alpha :: core :: logger（实现）
// =============================================================================
#include "trt_alpha/core/logger.hpp"

#include <iomanip>
#include <sstream>
#include <string>

namespace trt_alpha::core::detail {
namespace {

constexpr int kBoxWidth = 68;

//! 把一行内容放到 `| ... |` 里，右侧填充空格到 kBoxWidth。
void appendLine(std::ostringstream& oss, const std::string& content)
{
    const int innerWidth = kBoxWidth - 2;   // 去掉两侧 "| " 和 " |"
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
    const std::size_t elemSize = sizeOf(info.dtype);
    const double mb = static_cast<double>(info.bytes) / (1024.0 * 1024.0);

    std::ostringstream oss;
    oss << "+" << std::string(kBoxWidth - 2, '-') << "+\n";

    {
        std::ostringstream line;
        line << "[ALLOC] " << (info.name ? info.name : "");
        appendLine(oss, line.str());
    }
    {
        std::ostringstream line;
        line << "  batch    = " << info.batch;
        appendLine(oss, line.str());
    }
    {
        std::ostringstream line;
        line << "  channels = " << info.channels;
        appendLine(oss, line.str());
    }
    {
        std::ostringstream line;
        line << "  height   = " << info.height;
        appendLine(oss, line.str());
    }
    {
        std::ostringstream line;
        line << "  width    = " << info.width;
        appendLine(oss, line.str());
    }
    {
        std::ostringstream line;
        line << "  dtype    = " << nameOf(info.dtype)
             << " (" << elemSize << " bytes)";
        appendLine(oss, line.str());
    }
    {
        std::ostringstream line;
        line << "  calc     = " << info.batch << " * " << info.channels
             << " * " << info.height << " * " << info.width
             << " * " << elemSize;
        appendLine(oss, line.str());
    }
    {
        std::ostringstream line;
        line << "  bytes    = " << info.bytes;
        appendLine(oss, line.str());
    }
    {
        std::ostringstream line;
        line << "  MB       = " << formatDouble(mb, 2) << " MB";
        appendLine(oss, line.str());
    }
    {
        std::ostringstream line;
        line << "  space    = "
             << (info.space == MemorySpace::Device ? "Device" : "Host");
        appendLine(oss, line.str());
    }

    oss << "+" << std::string(kBoxWidth - 2, '-') << "+\n";

    // 一次性原子输出（与 TRT_LOG_* 宏共用同一把锁）
    std::lock_guard<std::mutex> lk(logMutex());
    std::cout << oss.str();
}

}  // namespace trt_alpha::core::detail