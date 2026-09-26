// =============================================================================
//  trt_alpha :: app :: commands（实现）
// =============================================================================
#include "options.hpp"

#include "trt_alpha/core/config.hpp"
#include "trt_alpha/core/inference_pool.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/core/paths.hpp"
#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/datasource/opencv_source.hpp"
#include "trt_alpha/datasource/source_config.hpp"
#include "trt_alpha/pipeline/pipeline.hpp"
#include "trt_alpha/pipeline/pipeline_config.hpp"
#include "trt_alpha/renderer/opencv_renderer.hpp"
#include "commands.hpp"

#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <iostream>

namespace trt_alpha::app {

void printUsage()
{
    std::cout <<
        "trt_alpha v1.0\n"
        "\n"
        "Usage:\n"
        "  trt_alpha list\n"
        "  trt_alpha run    --image <path> [--engine <trt>] [--save] [--show]\n"
        "  trt_alpha run    --video <path> [--engine <trt>] [--save]\n"
        "  trt_alpha bench  (TODO)\n"
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
        "  --workers <n>    inference pool workers (default: 0=auto)\n"
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

    // 1. 工程根覆盖（必须在首次 Paths::root() 之前）
    if (!opt.root.empty())
    {
        trt_alpha::core::Paths::setOverride(opt.root);
    }

    // 2. 读 INI
    const std::string iniPath =
        opt.config.empty() ? "configs/yolov8.ini" : opt.config;
    trt_alpha::core::ModelConfig modelCfg = trt_alpha::core::loadModelConfig(iniPath);

    // 3. 命令行覆盖
    if (!opt.engine.empty()) { modelCfg.engine = opt.engine; }
    if (opt.batch > 0)       { modelCfg.batchSize = opt.batch; }

    // 4. 类别
    modelCfg.classNames = trt_alpha::core::loadClassNamesFile(modelCfg.classNamesFile);

    // 5. 推理池
    trt_alpha::core::InferencePool pool(
        modelCfg,
        [name = opt.model]() -> std::unique_ptr<trt_alpha::IModel> {
            return trt_alpha::ModelRegistry::instance().create(name);
        },
        opt.workers);

    // 6. 数据源（v1.0 先只做 --image）
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

    // 7. 渲染
    trt_alpha::renderer::OpenCVRenderer renderer;

    // 8. Pipeline
    trt_alpha::pipeline::PipelineConfig cfg;
    cfg.sources = std::move(sources);
    cfg.pools = { &pool };
    cfg.renderer = &renderer;
    cfg.classNames = modelCfg.classNames;
    cfg.saveEnabled = opt.save;
    cfg.saveDir = opt.saveDir;
    cfg.showEnabled = opt.show;

    trt_alpha::pipeline::Pipeline p(std::move(cfg));
    p.start();
    p.waitForCompletion();

    return 0;
}

int benchCommand(const std::vector<std::string>& /*args*/)
{
    TRT_LOG_WARN("trt_alpha bench: not implemented yet");
    return 1;
}

int buildCommand(const std::vector<std::string>& /*args*/)
{
    TRT_LOG_WARN("trt_alpha build: not implemented yet");
    return 1;
}

}  // namespace trt_alpha::app