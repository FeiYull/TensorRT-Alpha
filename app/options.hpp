// =============================================================================
//  trt_alpha :: app :: options
// -----------------------------------------------------------------------------
//  CLI 参数结构体 + 解析。
// =============================================================================
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace trt_alpha::app {

//! `run` 命令的参数。
struct RunOptions
{
    // ---- 数据源 ----
    //! --source <path|url|id>：一个参数涵盖图片 / 目录 / 视频 / 网络流 / 摄像头 id。
    //! 具体类型由框架内部判别（datasource::resolveSource），命令行不再区分。
    //! 可重复出现：每个 --source 起一路独立数据源（多路视频 / 多路流）。
    std::vector<std::string> sources;

    //! --decode <opencv|ffmpeg|auto>：视频用哪个库解码。默认 opencv（保持现状）。
    //! 图片 / 目录 / 摄像头始终走 OpenCV。
    std::string decode;

    //! --hwaccel <auto|none|nvdec>：FFmpeg 后端是否用 NVDEC 硬解。默认 auto。
    std::string hwaccel;

    //! --host-copy <auto|on|off>：解码后是否把像素拷回主机。
    //! 空 = 用 INI [source] need_host_copy（默认 auto）。
    std::string hostCopy;

    // ---- 模型 ----
    std::string config;         //!< --config <ini>（不传则 configs/<net>.ini）
    std::string engine;         //!< --engine <trt>（覆盖 INI）
    std::string net = "yolov8"; //!< --net <name>（模型名，默认 yolov8）
    int batch = -1;             //!< --batch <n>（-1 = 用 INI）

    // ---- 渲染 ----
    bool show = false;
    bool save = false;
    std::string saveDir;        //!< --save [dir]：显式输出目录；空 = 默认 save/<net>
    std::string render = "auto"; //!< --render <auto|opencv|gl>：渲染后端（默认 auto）

    // ---- 池 ----
    int workers = -1;           //!< --workers <n>（<=0 = 用 INI [pool].workers）

    // ---- 全局 ----
    std::string root;           //!< --root <dir>

    //! 校验：至少一个源；数值合法。
    void validate() const;

    //! 是否指定了任何数据源。
    [[nodiscard]] bool hasSource() const noexcept;

    //! 解析 INI 路径：显式 --config 优先，否则 configs/<net>.ini。
    [[nodiscard]] std::string resolveConfigPath() const;
};

//! 解析 `run` 命令的参数（args[0] == "run"，跳过）。
RunOptions parseRunOptions(const std::vector<std::string>& args);

// =============================================================================
//  bench 命令
// =============================================================================

//! `bench` 命令的参数。
struct BenchOptions
{
    std::string engine;             //!< --engine <trt>（与 config 至少一个）
    std::string config;             //!< --config <ini>（不传则 configs/<net>.ini）
    std::string net = "yolov8";     //!< --net <name>
    int batch = -1;                 //!< --batch <n>（-1 = 用 INI）
    int iters = 100;                //!< --iters <n>（测量次数）
    int warmup = 10;                //!< --warmup <n>（预热次数，不计入统计）
    int srcW = 0;                   //!< --src <WxH> 的 W（0 = 用引擎输入尺寸）
    int srcH = 0;                   //!< --src <WxH> 的 H（0 = 用引擎输入尺寸）
    std::string root;               //!< --root <dir>

    void validate() const;

    //! 解析 INI 路径：显式 --config 优先，否则 configs/<net>.ini。
    [[nodiscard]] std::string resolveConfigPath() const;
};

BenchOptions parseBenchOptions(const std::vector<std::string>& args);

//! 打印用法。
void printUsage();

}  // namespace trt_alpha::app
