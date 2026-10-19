// =============================================================================
//  sample_yolov26_depth —— trt_alpha 使用示例（YOLOV26 单目深度估计）
// -----------------------------------------------------------------------------
//  演示如何用 trt_alpha 框架跑 YOLOV26 单目深度估计推理：
//    1. 手工构造 ModelConfig（不读 INI）
//    2. 构造 InferencePool（1 worker）
//    3. 构造 OpenCVSource（图片 / 视频）
//    4. 构造 OpenCVRenderer
//    5. 组装 Pipeline，起线程，等待完成
//
//  用法：
//    sample_yolov26_depth <engine.trt> <input>          # input = 图片 / 视频
//    sample_yolov26_depth <engine.trt> <input> --save   # 存盘
//
//  注意：batchSize 必须与引擎一致。本系列的引擎是静态 batch（导出 ONNX 时
//  写死的），所以这里给的是 2；换成静态 batch = 1 的引擎就改成 1。
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
        std::cout << "usage: sample_yolov26_depth <engine.trt> <input> [--save]\n"
                  << "  input : image / video path\n"
                  << "  --save: save result images to save/\n";
        return 1;
    }

    const std::string enginePath = argv[1];
    const std::string inputPath  = argv[2];
    const bool saveEnabled = (argc >= 4 && std::string(argv[3]) == "--save");

    try
    {
        // ---- 1. ModelConfig ----
        trt_alpha::core::ModelConfig cfg;
        cfg.engine = enginePath;
        cfg.batchSize = 2;
        cfg.dstH = 768;
        cfg.dstW = 768;
        cfg.classNamesFile = "data/classes/depth.txt";

        // 模型特有字段（YOLOV26 单目深度估计 自己读）


        // 类别信息（渲染用）
        cfg.classNames = trt_alpha::core::loadClassNamesFile(cfg.classNamesFile);

        // ---- 2. InferencePool ----
        trt_alpha::core::InferencePool pool(
            cfg,
            []() -> std::unique_ptr<trt_alpha::IModel> {
                return trt_alpha::ModelRegistry::instance().create("yolov26_depth");
            },
            /*workers=*/1);

        // ---- 3. 数据源 ----
        trt_alpha::datasource::SourceConfig srcCfg;
        srcCfg.batchSize = pool.resolvedBatch();
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

        // ---- 4. Renderer ----
        trt_alpha::renderer::OpenCVRenderer renderer;

        // ---- 5. Pipeline ----
        trt_alpha::pipeline::PipelineConfig pcfg;
        pcfg.sources = std::move(sources);
        pcfg.pools = { &pool };
        pcfg.renderer = &renderer;
        pcfg.classNames = cfg.classNames;
        pcfg.saveEnabled = saveEnabled;
        pcfg.saveDir = trt_alpha::core::Paths::resolveSaveDir("", "yolov26_depth");

        pcfg.showEnabled = false;

        trt_alpha::pipeline::Pipeline pipeline(std::move(pcfg));

        TRT_LOG_INFO("sample_yolov26_depth: starting pipeline");
        pipeline.start();
        pipeline.waitForCompletion();
        TRT_LOG_INFO("sample_yolov26_depth: done");

        return 0;
    }
    catch (const std::exception& e)
    {
        TRT_LOG_ERROR("sample_yolov26_depth: " << e.what());
        return 1;
    }
}
