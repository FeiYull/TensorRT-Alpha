// =============================================================================
//  trt_alpha :: app :: options（实现）
// =============================================================================
#include "options.hpp"

#include "trt_alpha/core/logger.hpp"

#include <stdexcept>
#include <string>

namespace trt_alpha::app {
namespace {

//! 取下一个参数，越界抛异常。
std::string nextArg(const std::vector<std::string>& args, std::size_t& i,
                    const std::string& flag)
{
    if (i + 1 >= args.size())
    {
        throw std::runtime_error("missing value for " + flag);
    }
    return args[++i];
}

//! 解析 int，非法抛异常。
int parseInt(const std::string& text, const std::string& flag)
{
    try
    {
        std::size_t consumed = 0;
        const int v = std::stoi(text, &consumed);
        if (consumed != text.size())
        {
            throw std::invalid_argument("trailing");
        }
        return v;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("invalid int for " + flag + ": '" + text + "'");
    }
}

//! 解析 "WxH"（大小写 x / X 均可），非法抛异常。
void parseSize(const std::string& text, const std::string& flag, int& w, int& h)
{
    const std::size_t sep = text.find_first_of("xX");
    if (sep == std::string::npos || sep == 0 || sep + 1 >= text.size())
    {
        throw std::runtime_error("invalid size for " + flag +
                                 ": '" + text + "' (expect WxH, e.g. 640x480)");
    }
    w = parseInt(text.substr(0, sep), flag);
    h = parseInt(text.substr(sep + 1), flag);
    if (w <= 0 || h <= 0)
    {
        throw std::runtime_error("invalid size for " + flag +
                                 ": '" + text + "' (W/H must be > 0)");
    }
}

}  // namespace

// =============================================================================
//  RunOptions
// =============================================================================

bool RunOptions::hasSource() const noexcept
{
    return !image.empty() || !images.empty() || !video.empty() || cameraId >= 0;
}

std::string RunOptions::resolveConfigPath() const
{
    if (!config.empty()) { return config; }
    return "configs/" + net + ".ini";
}

void RunOptions::validate() const
{
    const int sourceCount =
        (!image.empty()  ? 1 : 0) +
        (!images.empty() ? 1 : 0) +
        (!video.empty()  ? 1 : 0) +
        (cameraId >= 0   ? 1 : 0);

    if (sourceCount == 0)
    {
        throw std::runtime_error("run: no source specified "
                                 "(need one of --image / --images / --video / --camera)");
    }
    if (sourceCount > 1)
    {
        throw std::runtime_error("run: only one source allowed "
                                 "(got " + std::to_string(sourceCount) + ")");
    }
    if (batch == 0 || batch < -1)
    {
        throw std::runtime_error("run: --batch must be > 0 (got " +
                                 std::to_string(batch) + ")");
    }
    if (workers == 0 || workers < -1)
    {
        throw std::runtime_error("run: --workers must be > 0");
    }
    if (net.empty())
    {
        throw std::runtime_error("run: --net must not be empty");
    }
}

RunOptions parseRunOptions(const std::vector<std::string>& args)
{
    RunOptions opt;

    // args[0] == "run"，从 1 开始
    for (std::size_t i = 1; i < args.size(); ++i)
    {
        const std::string& a = args[i];

        if      (a == "--image")    { opt.image = nextArg(args, i, a); }
        else if (a == "--images")   { opt.images = nextArg(args, i, a); }
        else if (a == "--video")    { opt.video = nextArg(args, i, a); }
        else if (a == "--camera")   { opt.cameraId = parseInt(nextArg(args, i, a), a); }
        else if (a == "--config")   { opt.config = nextArg(args, i, a); }
        else if (a == "--engine")   { opt.engine = nextArg(args, i, a); }
        else if (a == "--net")      { opt.net = nextArg(args, i, a); }
        else if (a == "--batch")    { opt.batch = parseInt(nextArg(args, i, a), a); }
        else if (a == "--save-dir") { opt.saveDir = nextArg(args, i, a); }
        else if (a == "--workers")  { opt.workers = parseInt(nextArg(args, i, a), a); }
        else if (a == "--root")     { opt.root = nextArg(args, i, a); }
        else if (a == "--save")     { opt.save = true; }
        else if (a == "--show")     { opt.show = true; }
        else
        {
            throw std::runtime_error("run: unknown option '" + a + "'");
        }
    }

    return opt;
}

// =============================================================================
//  BenchOptions
// =============================================================================

std::string BenchOptions::resolveConfigPath() const
{
    if (!config.empty()) { return config; }
    return "configs/" + net + ".ini";
}

void BenchOptions::validate() const
{
    if (engine.empty() && config.empty() && net.empty())
    {
        throw std::runtime_error("bench: need --engine or --config or --net");
    }
    if (iters <= 0)
    {
        throw std::runtime_error("bench: --iters must be > 0 (got " +
                                 std::to_string(iters) + ")");
    }
    if (warmup < 0)
    {
        throw std::runtime_error("bench: --warmup must be >= 0 (got " +
                                 std::to_string(warmup) + ")");
    }
    if (batch == 0 || batch < -1)
    {
        throw std::runtime_error("bench: --batch must be > 0 (got " +
                                 std::to_string(batch) + ")");
    }
}

BenchOptions parseBenchOptions(const std::vector<std::string>& args)
{
    BenchOptions opt;

    // args[0] == "bench"，从 1 开始
    for (std::size_t i = 1; i < args.size(); ++i)
    {
        const std::string& a = args[i];

        if      (a == "--engine") { opt.engine = nextArg(args, i, a); }
        else if (a == "--config") { opt.config = nextArg(args, i, a); }
        else if (a == "--net")    { opt.net = nextArg(args, i, a); }
        else if (a == "--batch")  { opt.batch = parseInt(nextArg(args, i, a), a); }
        else if (a == "--iters")  { opt.iters = parseInt(nextArg(args, i, a), a); }
        else if (a == "--warmup") { opt.warmup = parseInt(nextArg(args, i, a), a); }
        else if (a == "--src")    { parseSize(nextArg(args, i, a), a, opt.srcW, opt.srcH); }
        else if (a == "--root")   { opt.root = nextArg(args, i, a); }
        else
        {
            throw std::runtime_error("bench: unknown option '" + a + "'");
        }
    }

    return opt;
}

}  // namespace trt_alpha::app