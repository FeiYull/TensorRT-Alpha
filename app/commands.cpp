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
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trt_alpha::app {
namespace {

namespace fs = std::filesystem;

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
        "trt_alpha v1.0 -- TensorRT inference framework\n"
        "\n"
        "Usage:\n"
        "  trt_alpha list\n"
        "  trt_alpha run   <source> [options]\n"
        "  trt_alpha bench <options>\n"
        "  trt_alpha build (TODO)\n"
        "\n"
        "Commands:\n"
        "  list   List all registered models\n"
        "  run    Run inference on image / images / video / camera\n"
        "  bench  Benchmark latency and throughput\n"
        "  build  Convert ONNX to engine (TODO)\n"
        "\n"
        "Run options:\n"
        "  --image <path>    single image (a directory is also accepted: scanned, one level)\n"
        "  --images <dir>    image directory (one level)\n"
        "  --video <path|url>  video file, or stream URL (rtsp / rtmp / http / https)\n"
        "  --camera <id>     camera device id\n"
        "  --net <name>      model name (default: yolov8)\n"
        "  --config <ini>    model INI (default: configs/<net>.ini)\n"
        "  --engine <trt>    override INI's engine path\n"
        "  --batch <n>       override INI's batch_size\n"
        "  --workers <n>     inference pool workers (default: INI [pool].workers)\n"
        "  --save [dir]      save result images (default dir: save/<net>)\n"
        "  --show            show result window\n"
        "  --root <dir>      override project root\n"
        "\n"
        "Bench options:\n"
        "  --net <name>      model name (default: yolov8)\n"
        "  --config <ini>    model INI (default: configs/<net>.ini)\n"
        "  --engine <trt>    override INI's engine path\n"
        "  --batch <n>       override INI's batch_size\n"
        "  --iters <n>       bench iterations (default: 100)\n"
        "  --warmup <n>      warmup iterations (default: 10)\n"
        "  --src <WxH>       source frame size (default: engine input size)\n"
        "  --root <dir>      override project root\n"
        "\n"
        "Examples:\n"
        "  trt_alpha list\n"
        "  trt_alpha run --image data/bus.jpg --net yolov8 --save\n"
        "  trt_alpha run --video data/people.mp4 --net yolor --show\n"
        "  trt_alpha run --video rtsp://192.168.1.10:554/stream1 --net yolov8 --show\n"
        "  trt_alpha run --camera 0 --net yolov8_pose --show\n"
        "  trt_alpha bench --net yolov8 --iters 100 --warmup 10\n"
        "  trt_alpha run --image data/bus.jpg --net yolov8 --engine D:/models/yolov8n.trt\n";
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

    // 读 INI（--config 优先，否则 configs/<net>.ini）
    const std::string iniPath = opt.resolveConfigPath();
    trt_alpha::core::ModelConfig modelCfg = trt_alpha::core::loadModelConfig(iniPath);

    // CLI 覆盖：值 / 来源 / 消费痕迹三者一起改。
    // 只改字段不改 extras 的话，框里会显示"ini 的旧值 + CLI 的来源"，自相矛盾。
    if (!opt.engine.empty())
    {
        modelCfg.engine = opt.engine;
        modelCfg.extras["model.engine"] = opt.engine;
        modelCfg.setOrigin("model.engine", "CLI");
    }
    if (opt.batch > 0)
    {
        modelCfg.batchSize = opt.batch;
        modelCfg.extras["input.batch_size"] = std::to_string(opt.batch);
        modelCfg.setOrigin("input.batch_size", "CLI");
    }

    modelCfg.classNames = trt_alpha::core::loadClassNamesFile(modelCfg.classNamesFile);

    // ---- 池 / 输出参数：CLI 优先，否则取 INI（取过就自动登记为"已消费"）----
    // workers：0 = 交给 InferencePool 自动（hardware_concurrency，上限 8）
    std::size_t workers = 0;
    if (opt.workers > 0)
    {
        workers = static_cast<std::size_t>(opt.workers);
        modelCfg.extras["pool.workers"] = std::to_string(opt.workers);
        modelCfg.setOrigin("pool.workers", "CLI");
        modelCfg.markRead("pool.workers");
    }
    else
    {
        const int w = modelCfg.getInt("pool.workers", 1);
        workers = (w > 0) ? static_cast<std::size_t>(w) : std::size_t{0};
    }

    // 队列无上限会吃光内存（相机源尤其危险）：<=0 一律退回框架默认值
    const int qTask   = modelCfg.getInt("pool.max_queue_size", 16);
    const int qResult = modelCfg.getInt("pool.result_queue_size", 32);

    // 存盘目录：CLI --save <dir> > ini output.save_dir > 默认 save/<net>。
    // 谁显式给了目录就用谁的（原样，不拼子目录）。base.ini 不再写死 save_dir，
    // 否则 --save 无路径时会被它顶成 "save"，拿不到 save/<net> 的默认结构。
    std::string saveDir = opt.saveDir;
    const bool saveDirFromCli = !saveDir.empty();
    if (saveDir.empty())
    {
        saveDir = modelCfg.getString("output.save_dir", "");
    }
    if (saveDir.empty())
    {
        saveDir = trt_alpha::core::Paths::resolveSaveDir("", opt.net);
        modelCfg.extras["output.save_dir"] = saveDir;
        modelCfg.setOrigin("output.save_dir", "default");
    }
    else if (saveDirFromCli)
    {
        modelCfg.extras["output.save_dir"] = saveDir;
        modelCfg.setOrigin("output.save_dir", "CLI");
    }
    modelCfg.markRead("output.save_dir");   // 框里显示最终生效目录，来源如实标注

    // 窗口名只有 --show 时才真正用到，但读一下即可见
    const std::string showWindow = modelCfg.getString("output.show_window", "trt_alpha");

    // 推理池
    trt_alpha::core::InferencePool pool(
        modelCfg,
        [name = opt.net]() -> std::unique_ptr<trt_alpha::IModel> {
            return trt_alpha::ModelRegistry::instance().create(name);
        },
        workers,
        (qTask > 0) ? static_cast<std::size_t>(qTask) : std::size_t{16});

    // 打印本次实际生效的配置（含引擎真相）。
    // 放在这里：模型已 init、还没开始推帧 —— 只要模型加载成功就能看到，
    // 不依赖是否真的跑出第一帧（摄像头打不开时也能看到配置）。
    // 用 pool.modelConfig()：模型 init 时的读取痕迹留在那一份上，
    // 用它才能正确标出 ini 里的死键（[unused]）。
    trt_alpha::core::logConfigBox(pool.modelConfig(), opt.net, iniPath,
                                  &pool.ioDesc(), pool.resolvedBatch());

    // 数据源
    trt_alpha::datasource::SourceConfig srcCfg;
    // batch 必须取【引擎解析后】的值，不能用 ini / CLI 的原值：
    // 静态引擎会忽略请求值、动态引擎越界直接报错 —— 若按原值攒批，
    // 数据源的批大小会与模型实际 batch 不一致 → 越界写。
    srcCfg.batchSize = pool.resolvedBatch();
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

    // 安全守卫：结果按"原文件名"存盘 ⇒ 输出目录 == 输入图片目录会覆盖原图。
    // 只有目录源（--image <目录> / --images）可能撞上；视频 / 相机 / 流无此风险。
    if (opt.save && !trt_alpha::core::Paths::isUrl(srcCfg.path))
    {
        std::error_code ec;
        const fs::path inPath = trt_alpha::core::Paths::resolve(srcCfg.path);
        if (fs::is_directory(inPath, ec))
        {
            std::error_code ec2;
            const fs::path inCanon  = fs::weakly_canonical(inPath, ec);
            const fs::path outCanon = fs::weakly_canonical(
                trt_alpha::core::Paths::resolve(saveDir), ec2);
            if (!ec && !ec2 && inCanon == outCanon)
            {
                throw std::runtime_error(
                    "run: save dir equals the input image dir (" +
                    trt_alpha::core::Paths::toDisplay(outCanon) +
                    "); results are named after the source files and would "
                    "overwrite the originals - pass another dir to --save");
            }
        }
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
    pcfg.resultQueueSize = (qResult > 0) ? static_cast<std::size_t>(qResult) : std::size_t{32};
    pcfg.saveEnabled = opt.save;
    pcfg.saveDir = saveDir;
    pcfg.showEnabled = opt.show;
    pcfg.showWindow = showWindow;

    trt_alpha::pipeline::Pipeline p(std::move(pcfg));
    p.start();
    p.waitForCompletion();

    // 线程内的错误没有异常出口（join 会吞掉），靠 Pipeline 记的失败标记报出来。
    // 否则"图超出引擎 profile / 批内分辨率不一致"这类错误只留一行 ERROR 日志，
    // 退出码却是 0 —— 脚本和 CI 会当成跑成功。
    if (p.failed())
    {
        TRT_LOG_ERROR("run: aborted, reason: " << p.firstError());
        return 1;
    }

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
    const std::string iniPath = opt.resolveConfigPath();
    trt_alpha::core::ModelConfig cfg = trt_alpha::core::loadModelConfig(iniPath);

    if (!opt.engine.empty()) { cfg.engine = opt.engine; }
    if (opt.batch > 0)       { cfg.batchSize = opt.batch; }

    cfg.classNames = trt_alpha::core::loadClassNamesFile(cfg.classNamesFile);

    // 造模型
    auto model = trt_alpha::ModelRegistry::instance().create(opt.net);
    model->init(cfg);

    // 固定输入：batch / H / W 一律取【模型实际生效的配置】
    //   - 传入的 cfg 是 const 引用，init 无法回写；引擎解析结果落在 model->config() 里。
    //   - H/W 默认 = 引擎声明的输入尺寸（与真实推理完全一致）。
    //   - 显式 --src WxH 时用该源帧尺寸，让 letterbox 也参与计时（更接近相机帧）。
    const trt_alpha::core::ModelConfig& rc = model->config();
    const int B = rc.batchSize;
    const int W = (opt.srcW > 0) ? opt.srcW : rc.dstW;
    const int H = (opt.srcH > 0) ? opt.srcH : rc.dstH;
    if (W <= 0 || H <= 0)
    {
        throw std::runtime_error("bench: cannot determine input size (engine has dynamic "
                                 "H/W and no --src given); pass --src WxH");
    }
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

    std::cout << "=== bench: " << opt.net << " ===\n";
    std::cout << "engine  : " << cfg.engine << "\n";
    std::cout << "batch   : " << B << "\n";
    std::cout << "input   : " << W << "x" << H
              << (opt.srcW > 0 ? " (--src)" : " (engine)") << "\n";
    std::cout << "iters   : " << opt.iters << " (warmup " << opt.warmup << ")\n";
    std::cout << "\n";
    std::cout << "Latency (ms, per batch of " << B << "):\n";
    std::cout << "  mean : " << mean << "\n";
    std::cout << "  p50  : " << pct(0.50) << "\n";
    std::cout << "  p90  : " << pct(0.90) << "\n";
    std::cout << "  p99  : " << pct(0.99) << "\n";
    std::cout << "  min  : " << lat.front() << "\n";
    std::cout << "  max  : " << lat.back() << "\n";
    std::cout << "\n";
    std::cout << "Throughput:\n";
    // 吞吐按【每批耗时】换算成【每秒张数】：一批 B 张，一次迭代平均 mean 毫秒。
    // （原先漏乘 B，batch>1 时会被低估 B 倍。）
    std::cout << "  FPS  : " << (static_cast<double>(B) * 1000.0 / mean)
              << "  (images/s)\n";
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