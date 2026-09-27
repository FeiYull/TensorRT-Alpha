// =============================================================================
//  trt_alpha :: app :: commands（实现）
// =============================================================================
#include "commands.hpp"
#include "options.hpp"

#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/config.hpp"
#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/core/inference_pool.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/core/paths.hpp"
#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/datasource/opencv_source.hpp"
#include "trt_alpha/datasource/source_config.hpp"
#include "trt_alpha/pipeline/pipeline.hpp"
#include "trt_alpha/pipeline/pipeline_config.hpp"
#include "trt_alpha/renderer/opencv_renderer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace trt_alpha::app {
namespace {

//! 造一个固定输入（灰色），用于 bench。尺寸 = 网络输入尺寸。
trt_alpha::core::Batch makeBenchBatch(int batchSize, int width, int height)
{
    auto buf = trt_alpha::core::Buffer::createHost(
        width * batchSize, height, 3, trt_alpha::core::DataType::UInt8);
    std::memset(buf->mutableData(), 128, buf->byteSize());

    trt_alpha::core::Batch b;
    b.sourceId = 0;
    b.firstFrameIndex = 0;
    b.buffer = buf;
    b.validCount = batchSize;

    const std::size_t oneFrame = static_cast<std::size_t>(width) * height * 3;
    for (int i = 0; i < batchSize; ++i)
    {
        trt_alpha::core::BufferView v;
        v.data = buf->data() + static_cast<std::size_t>(i) * oneFrame;
        v.width = width;
        v.height = height;
        v.stride = width * 3;
        v.channels = 3;
        v.dtype = trt_alpha::core::DataType::UInt8;
        v.space = trt_alpha::core::MemorySpace::Host;
        b.views.push_back(v);
    }
    return b;
}

}  // namespace

void printUsage()
{
    std::cout <<
        "trt_alpha v1.0\n"
        "\n"
        "Usage:\n"
        "  trt_alpha list\n"
        "  trt_alpha run    --image <path> [--engine <trt>] [--save] [--show]\n"
        "  trt_alpha run    --video <path> [--engine <trt>] [--save]\n"
        "  trt_alpha bench  --engine <trt> [--batch <n>] [--iters <n>] [--warmup <n>]\n"
        "  trt_alpha build  (TODO)\n"
        "\n"
        "Options:\n"
        "  --config <ini>   model INI (default: configs/yolov8.ini)\n"
        "  --engine <trt>   override INI's engine path\n"
        "  --model <name>   model name (default: yolov8)\n"
        "  --batch <n>      override INI's batch_size\n"
        "  --save           save result images\n"
        "  --save-dir <dir> output dir (default: save)\n"
        "  --show           show result window\n"
        "  --workers <n>    inference pool workers (default: 1)\n"
        "  --iters <n>      bench iterations (default: 100)\n"
        "  --warmup <n>     bench warmup iterations (default: 10)\n"
        "  --root <dir>     override project root\n";
}

int listCommand(const std::vector<std::string>& /*args*/)
{
    const auto names = trt_alpha::ModelRegistry::instance().names();

    std::cout << "Registered models (" << names.size() << "):\n";
    for (const auto& n : names)
    {
        std::cout << "  " << n << "\n";
    }

    return 0;
}

int runCommand(const std::vector<std::string>& args)
{
    const RunOptions opt = parseRunOptions(args);
    opt.validate();

    if (!opt.root.empty())
    {
        trt_alpha::core::Paths::setOverride(opt.root);
    }

    // 读 INI
    const std::string iniPath =
        opt.config.empty() ? "configs/yolov8.ini" : opt.config;
    trt_alpha::core::ModelConfig modelCfg = trt_alpha::core::loadModelConfig(iniPath);

    if (!opt.engine.empty()) { modelCfg.engine = opt.engine; }
    if (opt.batch > 0)       { modelCfg.batchSize = opt.batch; }

    modelCfg.classNames = trt_alpha::core::loadClassNamesFile(modelCfg.classNamesFile);

    // 推理池
    trt_alpha::core::InferencePool pool(
        modelCfg,
        [name = opt.model]() -> std::unique_ptr<trt_alpha::IModel> {
            return trt_alpha::ModelRegistry::instance().create(name);
        },
        opt.workers);

    // 数据源
    trt_alpha::datasource::SourceConfig srcCfg;
    srcCfg.batchSize = modelCfg.batchSize;
    srcCfg.sourceId = 0;

    if (!opt.image.empty())
    {
        srcCfg.type = trt_alpha::datasource::SourceType::Image;
        srcCfg.path = opt.image;
    }
    else if (!opt.images.empty())
    {
        srcCfg.type = trt_alpha::datasource::SourceType::Images;
        srcCfg.path = opt.images;
    }
    else if (!opt.video.empty())
    {
        srcCfg.type = trt_alpha::datasource::SourceType::Video;
        srcCfg.path = opt.video;
    }
    else if (opt.cameraId >= 0)
    {
        srcCfg.type = trt_alpha::datasource::SourceType::Camera;
        srcCfg.cameraId = opt.cameraId;
    }
    else
    {
        throw std::runtime_error("run: no source (should not happen)");
    }

    std::vector<std::unique_ptr<trt_alpha::datasource::IDataSource>> sources;
    sources.push_back(
        std::make_unique<trt_alpha::datasource::OpenCVSource>(srcCfg));

    // 渲染
    trt_alpha::renderer::OpenCVRenderer renderer;

    // Pipeline
    trt_alpha::pipeline::PipelineConfig pcfg;
    pcfg.sources = std::move(sources);
    pcfg.pools = { &pool };
    pcfg.renderer = &renderer;
    pcfg.classNames = modelCfg.classNames;
    pcfg.saveEnabled = opt.save;
    pcfg.saveDir = opt.saveDir;
    pcfg.showEnabled = opt.show;

    trt_alpha::pipeline::Pipeline p(std::move(pcfg));
    p.start();
    p.waitForCompletion();

    return 0;
}

int benchCommand(const std::vector<std::string>& args)
{
    const BenchOptions opt = parseBenchOptions(args);
    opt.validate();

    if (!opt.root.empty())
    {
        trt_alpha::core::Paths::setOverride(opt.root);
    }

    // 读 INI
    const std::string iniPath =
        opt.config.empty() ? "configs/yolov8.ini" : opt.config;
    trt_alpha::core::ModelConfig cfg = trt_alpha::core::loadModelConfig(iniPath);

    if (!opt.engine.empty()) { cfg.engine = opt.engine; }
    if (opt.batch > 0)       { cfg.batchSize = opt.batch; }

    cfg.classNames = trt_alpha::core::loadClassNamesFile(cfg.classNamesFile);

    // 造模型
    auto model = trt_alpha::ModelRegistry::instance().create(opt.model);
    model->init(cfg);

    // 固定输入
    const int W = cfg.dstW;
    const int H = cfg.dstH;
    const int B = cfg.batchSize;
    trt_alpha::core::Batch batch = makeBenchBatch(B, W, H);

    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // 预热
    for (int i = 0; i < opt.warmup; ++i)
    {
        model->setBatch(batch);
        model->preprocess();
        model->infer();
        model->postprocess();
        model->reset();
    }

    // 测量
    std::vector<double> lat;
    lat.reserve(static_cast<std::size_t>(opt.iters));

    double sumSetBatch = 0.0;
    double sumPre      = 0.0;
    double sumInfer    = 0.0;
    double sumPost     = 0.0;

    for (int i = 0; i < opt.iters; ++i)
    {
        const auto t0 = now();
        model->setBatch(batch);
        const auto t1 = now();
        model->preprocess();
        const auto t2 = now();
        model->infer();
        const auto t3 = now();
        model->postprocess();
        const auto t4 = now();

        sumSetBatch += ms(t0, t1);
        sumPre      += ms(t1, t2);
        sumInfer    += ms(t2, t3);
        sumPost     += ms(t3, t4);
        lat.push_back(ms(t0, t4));

        model->reset();
    }

    // 统计
    std::sort(lat.begin(), lat.end());
    const std::size_t n = lat.size();
    const auto pct = [&](double p) {
        const std::size_t i = static_cast<std::size_t>(p * static_cast<double>(n - 1));
        return lat[i];
    };
    double sum = 0.0;
    for (double v : lat) { sum += v; }
    const double mean = sum / static_cast<double>(n);

    std::cout << "=== bench: " << opt.model << " ===\n";
    std::cout << "engine  : " << cfg.engine << "\n";
    std::cout << "batch   : " << B << "\n";
    std::cout << "iters   : " << opt.iters << " (warmup " << opt.warmup << ")\n";
    std::cout << "\n";
    std::cout << "Latency (ms):\n";
    std::cout << "  mean : " << mean << "\n";
    std::cout << "  p50  : " << pct(0.50) << "\n";
    std::cout << "  p90  : " << pct(0.90) << "\n";
    std::cout << "  p99  : " << pct(0.99) << "\n";
    std::cout << "  min  : " << lat.front() << "\n";
    std::cout << "  max  : " << lat.back() << "\n";
    std::cout << "\n";
    std::cout << "Throughput:\n";
    std::cout << "  FPS  : " << (1000.0 / mean) << "\n";
    std::cout << "\n";
    std::cout << "Per-step (mean, ms):\n";
    std::cout << "  setBatch    : " << (sumSetBatch / static_cast<double>(n)) << "\n";
    std::cout << "  preprocess  : " << (sumPre      / static_cast<double>(n)) << "\n";
    std::cout << "  infer       : " << (sumInfer    / static_cast<double>(n)) << "\n";
    std::cout << "  postprocess : " << (sumPost     / static_cast<double>(n)) << "\n";

    return 0;
}

int buildCommand(const std::vector<std::string>& /*args*/)
{
    TRT_LOG_ERROR("trt_alpha build: not implemented yet");
    return 1;
}

}  // namespace trt_alpha::app