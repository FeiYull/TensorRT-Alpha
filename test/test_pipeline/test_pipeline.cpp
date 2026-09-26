// =============================================================================
//  test/test_pipeline/test_pipeline.cpp
// -----------------------------------------------------------------------------
//  Pipeline 集成测试（真推理 + 详细日志）：
//    [1] 图片 / 视频 + YOLOv8 + 渲染 → 结果存盘
//    [2] 输出诊断信息（views / detections / validCount）
//
//  用法：
//    test_pipeline <engine.trt> <image_or_video>
// =============================================================================
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/core/paths.hpp"
#include "trt_alpha/datasource/opencv_source.hpp"
#include "trt_alpha/datasource/source_config.hpp"
#include "trt_alpha/core/inference_pool.hpp"
#include "trt_alpha/pipeline/pipeline.hpp"
#include "trt_alpha/pipeline/pipeline_config.hpp"
#include "trt_alpha/renderer/opencv_renderer.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;

using trt_alpha::core::InferencePool;
using trt_alpha::core::ModelConfig;
using trt_alpha::ModelRegistry;
using trt_alpha::datasource::OpenCVSource;
using trt_alpha::datasource::SourceConfig;
using trt_alpha::datasource::SourceType;
using trt_alpha::pipeline::Pipeline;
using trt_alpha::pipeline::PipelineConfig;
using trt_alpha::renderer::OpenCVRenderer;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

void step(const char* msg)
{
    std::cout << "[STEP] " << msg << std::endl;   // 强制 flush
}

}  // namespace

int main(int argc, char** argv)
{
    std::cout << "=== Pipeline tests ===\n";

    if (argc < 3)
    {
        std::cout << "usage: test_pipeline <engine.trt> <image_or_video>\n";
        std::cout << "(skipped: no engine+input args)\n";
        return 0;
    }

    const std::string enginePath = argv[1];
    const std::string inputPath = argv[2];

    std::cout << "engine : " << enginePath << "\n";
    std::cout << "input  : " << inputPath << "\n";

    try
    {
        // ---- 1. 模型配置 ----
        step("1. build ModelConfig");
        ModelConfig modelCfg;
        modelCfg.engine = enginePath;
        modelCfg.batchSize = 1;
        modelCfg.dstH = 640;
        modelCfg.dstW = 640;
        modelCfg.extras["num_class"] = "80";
        modelCfg.extras["conf_thresh"] = "0.25";
        std::cout << "       engine=" << modelCfg.engine
                  << " batch=" << modelCfg.batchSize << "\n";

        // ---- 2. 推理池 ----
        step("2. create InferencePool (1 worker)");
        InferencePool pool(
            modelCfg,
            []() -> std::unique_ptr<trt_alpha::IModel> {
                return ModelRegistry::instance().create("yolov8");
            },
            /*workers=*/1,
            /*maxQueueSize=*/16);
        std::cout << "       pool.size() = " << pool.size() << "\n";

        // ---- 3. 数据源 ----
        step("3. create OpenCVSource");
        SourceConfig srcCfg;
        srcCfg.batchSize = 1;
        srcCfg.sourceId = 0;
        if (inputPath.find(".mp4") != std::string::npos ||
            inputPath.find(".avi") != std::string::npos ||
            inputPath.find(".mov") != std::string::npos)
        {
            srcCfg.type = SourceType::Video;
        }
        else
        {
            srcCfg.type = SourceType::Image;
        }
        srcCfg.path = inputPath;
        std::cout << "       type=" << (srcCfg.type == SourceType::Video ? "video" : "image")
                  << " batchSize=" << srcCfg.batchSize << "\n";

        std::vector<std::unique_ptr<trt_alpha::datasource::IDataSource>> sources;
        sources.push_back(std::make_unique<OpenCVSource>(srcCfg));
        std::cout << "       sources.size() = " << sources.size() << "\n";

        // ---- 4. 渲染器 ----
        step("4. create OpenCVRenderer");
        OpenCVRenderer renderer;

        // ---- 5. Pipeline 配置 ----
        step("5. build PipelineConfig");
        PipelineConfig cfg;
        cfg.sources = std::move(sources);
        cfg.pools = { &pool };
        cfg.sourceToPool = {};
        cfg.renderer = &renderer;
        cfg.resultQueueSize = 32;
        cfg.saveEnabled = true;
        cfg.saveDir = "test_pipeline_out";
        cfg.showEnabled = false;
        std::cout << "       sources=" << cfg.sources.size()
                  << " pools=" << cfg.pools.size()
                  << " sourceToPool=" << cfg.sourceToPool.size()
                  << " queueSize=" << cfg.resultQueueSize
                  << " saveDir=" << cfg.saveDir << "\n";

        // ---- 6. 跑 ----
        step("6. create Pipeline");
        const std::string outDir = "test_pipeline_out";
        std::error_code ec;
        fs::remove_all(outDir, ec);
        if (ec)
        {
            std::cout << "[WARN] cannot clean " << outDir << ": " << ec.message() << "\n";
        }

        Pipeline pipeline(std::move(cfg));
        std::cout << "       pipeline constructed\n";

        step("7. pipeline.start()");
        pipeline.start();
        std::cout << "       started, running=" << pipeline.running() << "\n";

        step("8. pipeline.waitForCompletion()");
        pipeline.waitForCompletion();
        std::cout << "       completed\n";

        // ---- 7. 检查结果 ----
        step("9. check output");
        check(fs::is_directory(outDir), "[1] output dir created");

        std::size_t count = 0;
        if (fs::is_directory(outDir))
        {
            for (const auto& e : fs::directory_iterator(outDir))
            {
                if (e.is_regular_file())
                {
                    ++count;
                }
            }
        }
        std::cout << "       output files: " << count << "\n";
        check(count > 0, "[2] at least one result image saved");
    }
    catch (const std::exception& e)
    {
        std::cout << "[FAIL] exception: " << e.what() << "\n";
        ++g_failures;
    }

    std::cout << "===================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}