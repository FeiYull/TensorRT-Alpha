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
    std::string config;         //!< --config <ini>（默认 configs/yolov8.ini）
    std::string engine;         //!< --engine <trt>（覆盖 INI）
    std::string model = "yolov8";
    int batch = -1;             //!< --batch <n>（-1 = 用 INI）

    // ---- 渲染 ----
    bool show = false;
    bool save = false;
    std::string saveDir = "save";

    // ---- 池 ----
    std::size_t workers = 1;    //!< 推理池 worker 数（默认 1）

    // ---- 全局 ----
    std::string root;           //!< --root <dir>

    //! 校验：只允许一个源；数值合法。
    void validate() const;

    //! 是否指定了任何数据源。
    [[nodiscard]] bool hasSource() const noexcept;
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
    std::string config;             //!< --config <ini>（默认 configs/yolov8.ini）
    std::string model = "yolov8";   //!< --model <name>
    int batch = -1;                 //!< --batch <n>（-1 = 用 INI）
    int iters = 100;                //!< --iters <n>（测量次数）
    int warmup = 10;                //!< --warmup <n>（预热次数，不计入统计）
    std::string root;               //!< --root <dir>

    void validate() const;
};

BenchOptions parseBenchOptions(const std::vector<std::string>& args);

//! 打印用法。
void printUsage();

}  // namespace trt_alpha::app