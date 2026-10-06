// =============================================================================
//  sample_u2net —— trt_alpha 使用示例（显著性分割）
// -----------------------------------------------------------------------------
//  用法：
//    sample_u2net <engine.trt> <input>          # input = 图片 / 视频
//    sample_u2net <engine.trt> <input> --save   # 存盘
// =============================================================================
#include "trt_alpha/core/config.hpp"
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

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    trt_alpha::core::installCrashHandler();

    if (argc < 3)
    {
        std::cout << "usage: sample_u2net <engine.trt> <input> [--save]\n";
        return 1;
    }

    const std::string enginePath = argv[1];
    const std::string inputPath  = argv[2];
    const bool saveEnabled = (argc >= 4 && std::string(argv[3]) == "--save");

    try
    {
        trt_alpha::core::ModelConfig cfg;
        cfg.engine = enginePath;
        cfg.batchSize = 1;
        cfg.dstH = 320;
        cfg.dstW = 320;
        cfg.classNamesFile = "data/classes/salient.txt";

        cfg.extras["num_class"] = "1";
        cfg.extras["norm_scale"] = "1.0";
        cfg.extras["mean"] = "0.485,0.456,0.406";
        cfg.extras["std"] = "0.229,0.224,0.225";
        cfg.extras["post_scale"] = "255.0";

        cfg.classNames = trt_alpha::core::loadClassNamesFile(cfg.classNamesFile);

        trt_alpha::core::InferencePool pool(
            cfg,
            []() -> std::unique_ptr<trt_alpha::IModel> {
                return trt_alpha::ModelRegistry::instance().create("u2net");
            },
            /*workers=*/1);

        trt_alpha::datasource::SourceConfig srcCfg;
        srcCfg.batchSize = cfg.batchSize;
        srcCfg.sourceId = 0;

        const bool isVideo =
            inputPath.find(".mp4") != std::string::npos ||
            inputPath.find(".avi") != std::string::npos ||
            inputPath.find(".mov") != std::string::npos;

        if (isVideo)
        {
            srcCfg.type = trt_alpha::datasource::SourceType::Video;
        }
        else
        {
            srcCfg.type = trt_alpha::datasource::SourceType::Image;
        }
        srcCfg.path = inputPath;

        std::vector<std::unique_ptr<trt_alpha::datasource::IDataSource>> sources;
        sources.push_back(
            std::make_unique<trt_alpha::datasource::OpenCVSource>(srcCfg));

        trt_alpha::renderer::OpenCVRenderer renderer;

        trt_alpha::pipeline::PipelineConfig pcfg;
        pcfg.sources = std::move(sources);
        pcfg.pools = { &pool };
        pcfg.renderer = &renderer;
        pcfg.classNames = cfg.classNames;
        pcfg.saveEnabled = saveEnabled;
        pcfg.saveDir = trt_alpha::core::Paths::resolveSaveDir("", "u2net");
        pcfg.showEnabled = false;

        trt_alpha::pipeline::Pipeline pipeline(std::move(pcfg));

        TRT_LOG_INFO("sample_u2net: starting pipeline");
        pipeline.start();
        pipeline.waitForCompletion();
        TRT_LOG_INFO("sample_u2net: done");

        return 0;
    }
    catch (const std::exception& e)
    {
        TRT_LOG_ERROR("sample_u2net: " << e.what());
        return 1;
    }
}