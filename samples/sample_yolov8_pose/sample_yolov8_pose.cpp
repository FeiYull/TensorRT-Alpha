// =============================================================================
//  sample_yolov8_pose —— trt_alpha 使用示例（姿态估计）
// -----------------------------------------------------------------------------
//  用法：
//    sample_yolov8_pose <engine.trt> <input>          # input = 图片 / 视频
//    sample_yolov8_pose <engine.trt> <input> --save   # 存盘
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
        std::cout << "usage: sample_yolov8_pose <engine.trt> <input> [--save]\n";
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
        cfg.dstH = 640;
        cfg.dstW = 640;
        cfg.classNamesFile = "data/classes/person.txt";

        cfg.extras["num_kpts"] = "17";
        cfg.extras["conf_thresh"] = "0.25";
        cfg.extras["iou_thresh"] = "0.7";
        cfg.extras["top_k"] = "300";

        cfg.classNames = trt_alpha::core::loadClassNamesFile(cfg.classNamesFile);

        trt_alpha::core::InferencePool pool(
            cfg,
            []() -> std::unique_ptr<trt_alpha::IModel> {
                return trt_alpha::ModelRegistry::instance().create("yolov8_pose");
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
        pcfg.saveDir = trt_alpha::core::Paths::resolveSaveDir("", "yolov8_pose");
        pcfg.showEnabled = false;

        trt_alpha::pipeline::Pipeline pipeline(std::move(pcfg));

        TRT_LOG_INFO("sample_yolov8_pose: starting pipeline");
        pipeline.start();
        pipeline.waitForCompletion();
        TRT_LOG_INFO("sample_yolov8_pose: done");

        return 0;
    }
    catch (const std::exception& e)
    {
        TRT_LOG_ERROR("sample_yolov8_pose: " << e.what());
        return 1;
    }
}