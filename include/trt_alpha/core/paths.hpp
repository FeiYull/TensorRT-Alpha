// =============================================================================
//  trt_alpha :: core :: paths —— 工程根目录识别 + 相对路径解析
// -----------------------------------------------------------------------------
//  目的：让程序在【任意工作目录】下都能找到 configs/ 与 data/ 下的默认资源，
//  既不需要 cd 到工程根，也不需要改代码里的路径字符串。
//
//  根目录探测顺序（第一个命中即采用，结果缓存一次）：
//      1. --root <dir>（外部显式指定，调 setOverride）
//      2. 环境变量 TRT_ALPHA_ROOT
//      3. 可执行文件所在目录逐级向上找"根标志"
//      4. 编译期 TRT_ALPHA_ROOT_DIR（CMake 写入的源码根，便于 out-of-tree 构建）
//      5. 当前工作目录逐级向上找"根标志"
//      6. 兜底 = 当前工作目录
//
//  "根标志" = 目录下存在 configs/ 子目录，或存在 .trt_alpha_root 文件
//  （后者给"发布时只拷 exe + data"的场景留的手工标记）。
//
//  Windows / Linux 通用：exe 定位分别用 GetModuleFileNameW 与 /proc/self/exe；
//  路径一律以 std::filesystem::path 承载，避免窄字符流在非 ASCII 路径上出错。
// =============================================================================
#pragma once

#include <filesystem>
#include <string>

namespace trt_alpha::core {

//! 全局路径策略（纯静态，无实例）。
//! 所有状态在首次 root() 时初始化一次，之后只读（除非显式 setOverride）。
class Paths
{
public:
    //! 显式指定根目录（来自 --root）。
    //! 必须在首次 root() 之前调用，否则抛 std::logic_error。
    //! 传空字符串 = 清空 override（回到自动探测）。
    static void setOverride(const std::string& dir);

    //! 工程根目录（绝对路径）。首次调用时探测并缓存。
    [[nodiscard]] static const std::filesystem::path& root();

    //! 根目录的来源说明（日志/报错用，如 "executable location"）。
    [[nodiscard]] static std::string rootSource();

    //! 字符串 → 路径。
    //! 非 ASCII 时兼容两种来源：命令行（cmd 的 ANSI 代码页）与配置文件（UTF-8），
    //! 按"哪个真实存在"择一，都不存在则按原生编码处理。
    [[nodiscard]] static std::filesystem::path toPath(const std::string& text);

    //! 相对路径以 root() 为基准展开；绝对路径原样返回；空串返回空路径。
    [[nodiscard]] static std::filesystem::path resolve(const std::string& text);

    //! 是否为网络 URL（形如 <scheme>://<host>...），如 rtsp / rtmp / http(s) / udp。
    //! 判定依据是 RFC 3986 的 scheme 形态（字母开头 + [A-Za-z0-9+.-]，且至少 2 字符），
    //! 因此 Windows 盘符 "D:/a.jpg" 与 "C://x" 都不会被误判。
    //! 用途：URL 不是本地文件 —— 必须跳过 requireFile / 目录扫描这类本地校验。
    [[nodiscard]] static bool isUrl(const std::string& text);

    //! resolve() + 存在性校验。缺失时抛出带"根目录来源 + 修复建议"的可读错误。
    //! role 用于错误消息（如 "config file" / "input image"）。
    [[nodiscard]] static std::filesystem::path requireFile(const std::string& text,
                                                          const std::string& role);

    //! 把绝对路径转成便于打印/日志的字符串（Windows 下为原生编码）。
    [[nodiscard]] static std::string toDisplay(const std::filesystem::path& path);

    //! 解析"结果存盘目录"。规则一句话：谁显式给了目录就用谁的（原样，不再拼子目录）；
    //! 都没给才回退 <默认根>/<模型名>（默认根 = "save"）。
    //!   resolveSaveDir("out", "yolov8") -> "out"
    //!   resolveSaveDir("",    "yolov8") -> "save/yolov8"
    //!   resolveSaveDir("",    "")       -> "save"
    [[nodiscard]] static std::string resolveSaveDir(const std::string& dir,
                                                    const std::string& modelName);
};

}  // namespace trt_alpha::core