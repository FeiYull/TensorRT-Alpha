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
    // ---- 数据源（四选一）----
    std::string image;          //!< --image <path>
    std::string images;         //!< --images <dir>
    std::string video;          //!< --video <path>
    int cameraId = -1;          //!< --camera <id>

    // ---- 模型 ----
    std::string config;         //!< --config <ini>（不传则 configs/<net>.ini）
    std::string engine;         //!< --engine <trt>（覆盖 INI）
    std::string net = "yolov8"; //!< --net <name>（模型名，默认 yolov8）
    int batch = -1;             //!< --batch <n>（-1 = 用 INI）

    // ---- 渲染 ----
    bool show = false;
    bool save = false;
    std::string saveDir;        //!< --save-dir <dir>（空 = 用 INI [output].save_dir）

    // ---- 池 ----
    int workers = -1;           //!< --workers <n>（<=0 = 用 INI [pool].workers）

    // ---- 全局 ----
    std::string root;           //!< --root <dir>

    //! 校验：只允许一个源；数值合法。
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