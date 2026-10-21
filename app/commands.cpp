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
#include "trt_alpha/datasource/source_factory.hpp"
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
        "  run    Run inference on an image / directory / video / stream / camera\n"
        "  bench  Benchmark latency and throughput\n"
        "  build  Convert ONNX to engine (TODO)\n"
        "\n"
        "Run options:\n"
        "  --source <path|url|id>\n"
        "                    the input source. Its kind is detected automatically:\n"
        "                      data/bus.jpg         -> image\n"
        "                      data/                -> image directory (one level)\n"
        "                      data/people.mp4      -> local video file\n"
        "                      rtsp://host:554/s1   -> network stream (rtsp/rtmp/http(s)/udp)\n"
        "                      0                    -> camera device id\n"
        "  --net <name>      model name (default: yolov8)\n"
        "  --config <ini>    model INI (default: configs/<net>.ini)\n"
        "  --engine <trt>    override INI's engine path\n"
        "  --batch <n>       override INI's batch_size\n"
        "  --workers <n>     inference pool workers (default: INI [pool].workers)\n"
        "  --save [dir]      save result images (default dir: save/<net>)\n"
        "  --show            show result window\n"
        "  --decode <name>   video decode backend: opencv (default) | ffmpeg | auto\n"
        "                    (ffmpeg needs a build with TRT_ALPHA_WITH_FFMPEG)\n"
        "  --hwaccel <mode>  FFmpeg only: auto (default) | none | nvdec\n"
        "  --host-copy <mode>\n"
        "                    auto (default) | on | off: copy decoded pixels back\n"
        "                    to the host. off keeps them on the device and is\n"
        "                    rejected when the run also renders or the model has\n"
        "                    no device-pixel path.\n"
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
        "  trt_alpha run --source data/bus.jpg --net yolov8 --save\n"
        "  trt_alpha run --source data/ --net yolov8 --save\n"
        "  trt_alpha run --source data/people.mp4 --net yolor --show\n"
        "  trt_alpha run --source rtsp://192.168.1.10:554/stream1 --net yolov8 --show\n"
        "  trt_alpha run --source 0 --net yolov8_pose --show\n"
        "  trt_alpha bench --net yolov8 --iters 100 --warmup 10\n"
        "  trt_alpha run --source data/bus.jpg --net yolov8 --engine D:/models/yolov8n.trt\n";
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

    // 渲染线程数：0 / 不写 = auto（Pipeline 在"只落盘"时按 CPU 核数取 1~8，
    // --show / exposeProcessed 一律强制 1）。负数由 Pipeline 兜底改回 auto。
    const int rThreads = modelCfg.getInt("pool.render_threads", 0);

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

    // 窗口名只有 --show 时才真正用到。默认带注册名 —— 同时开两个模型时窗口
    // 不会互相顶掉（imshow 会复用同名窗口）。ini 显式写了就原样使用。
    const std::string showWindow =
        modelCfg.getString("output.show_window", "trt_alpha:" + opt.net);

    // 渲染选项：按模型（<model>.ini 的 [render] 节），不放 base —— pose 的
    // 关键点阈值、seg 的掩码透明度，不同任务默认值不同。在 pool 构造前读，
    // 这样 markRead 痕迹会随 modelCfg 一起拷进 pool 的副本，框里才能如实标
    // "已消费"（而不是 [unused]）。
    trt_alpha::renderer::RenderOptions renderOptions;
    renderOptions.kptConfThresh =
        modelCfg.getFloat("render.kpt_conf_thresh", renderOptions.kptConfThresh);
    renderOptions.maskAlpha =
        modelCfg.getFloat("render.mask_alpha", renderOptions.maskAlpha);

    // JPEG 质量（只影响 --save 落盘）：1..100，默认 95 = OpenCV 默认值。
    // 必须在这里就读 —— 配置框在本函数稍后才打印，晚读会被误标成 [unused]。
    const int jpegQuality = modelCfg.getInt("output.jpeg_quality", 95);

    // [source] 段的解码选项。优先级：CLI > ini > 代码默认。
    // 同样必须赶在配置框之前读（pool 构造时会把 modelCfg 连同 readKeys 拷一份），
    // 否则 ini 里写了的键会被标成 [unused]，看着像没生效。
    const std::string decodeName = opt.decode.empty()
                                       ? modelCfg.getString("source.decode", "opencv")
                                       : opt.decode;
    const std::string hwaccelName = opt.hwaccel.empty()
                                        ? modelCfg.getString("source.hwaccel", "auto")
                                        : opt.hwaccel;
    const std::string hostCopyName =
        opt.hostCopy.empty() ? modelCfg.getString("source.need_host_copy", "auto")
                             : opt.hostCopy;
    const int openTimeoutMs = modelCfg.getInt("source.open_timeout_ms", 5000);
    const int readTimeoutMs = modelCfg.getInt("source.read_timeout_ms", 5000);

    // 立即解析：拼写错误要在这里就炸，而不是等到打开视频才发现。
    const trt_alpha::datasource::DecodeBackend decodeBackend =
        trt_alpha::datasource::requireDecodeBackend(decodeName);
    const trt_alpha::datasource::HwAccel hwAccel =
        trt_alpha::datasource::requireHwAccel(hwaccelName);
    const trt_alpha::datasource::HostCopy hostCopy =
        trt_alpha::datasource::requireHostCopy(hostCopyName);

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

    // 一个 --source 涵盖图片 / 目录 / 视频 / 网络流 / 摄像头 id；类型由
    // datasource::resolveSource 统一判别（与高层 Infer 共用同一套规则），
    // 命令行不再需要区分。目录会在源内部自动升级为 Images。
    {
        const trt_alpha::datasource::ResolvedSource rs =
            trt_alpha::datasource::resolveSource(opt.source);
        srcCfg.type     = rs.type;
        srcCfg.path     = rs.path;
        srcCfg.cameraId = rs.cameraId;
    }

    // 解码后端：取值已在函数前半段按 CLI > ini > 默认 的顺序定好并解析过
    // （那一步必须赶在配置框之前，见 decodeBackend / hwAccel）。
    srcCfg.decodeBackend = decodeBackend;
    srcCfg.hwAccel = hwAccel;
    srcCfg.openTimeoutMs = openTimeoutMs;
    srcCfg.readTimeoutMs = readTimeoutMs;

    // 安全守卫：结果按"原文件名"存盘 ⇒ 输出路径可能正好是输入文件本身。
    // 旧判据只处理"输入是目录"，漏掉了最常见的一种：输入是【单张图】而
    // --save 指向这张图所在目录（--source data/bus.jpg --save data）——
    // 此时输出 data/bus.jpg == 输入，原图被覆盖成画框版，不可逆。
    // 新判据：把每个输入文件按渲染器的落盘规则算出输出路径，与输入逐一比对。
    // 只有图片 / 图片目录源会产出"原文件名"，视频与相机天然无此风险。
    if (opt.save &&
        (srcCfg.type == trt_alpha::datasource::SourceType::Image ||
         srcCfg.type == trt_alpha::datasource::SourceType::Images) &&
        !trt_alpha::core::Paths::isUrl(srcCfg.path))
    {
        std::error_code ec;
        const fs::path inRoot  = trt_alpha::core::Paths::resolve(srcCfg.path);
        const fs::path outRoot = trt_alpha::core::Paths::resolve(saveDir);

        std::vector<fs::path> inputFiles;
        if (fs::is_directory(inRoot, ec))
        {
            for (const auto& entry : fs::directory_iterator(inRoot, ec))
            {
                if (entry.is_regular_file()) { inputFiles.push_back(entry.path()); }
            }
        }
        else if (fs::is_regular_file(inRoot, ec))
        {
            inputFiles.push_back(inRoot);
        }

        for (const fs::path& in : inputFiles)
        {
            const fs::path out = outRoot / (in.stem().string() + ".jpg");
            std::error_code ec1, ec2;
            const fs::path outCanon = fs::weakly_canonical(out, ec1);
            const fs::path inCanon  = fs::weakly_canonical(in,  ec2);
            if (!ec1 && !ec2 && outCanon == inCanon)
            {
                throw std::runtime_error(
                    "run: --save would overwrite the input image (" +
                    trt_alpha::core::Paths::toDisplay(inCanon) +
                    "); results are named after the source files - pass another "
                    "dir to --save");
            }
        }
    }

    // 渲染器：只在真的要存 / 要显时才挂。否则 Pipeline 会为每一帧白跑一遍
    // drawResult（画框 + 画字 + getTextSize）然后丢弃，纯浪费。
    // 同一件事也决定数据源要不要把像素搬回主机：没有渲染器就没人读主机帧，
    // NVDEC 路径可以整批留在显存里（1080p 每帧省 6.2 MB 的 D2H，约 0.475 ms，
    // 折合 22% 的帧预算）。
    const bool needRender = opt.save || opt.show;

    // 主机副本只在渲染器会读的时候才需要。但模型若还不认识 Batch::deviceBuffer，
    // 它照样从主机上传像素 —— 那种情况必须保留副本，否则它读到的是空指针。
    // 所以 auto 的条件是"要渲染 或 模型不支持设备像素"，只有两条都不成立才真的省。
    srcCfg.needHostCopy = needRender || !pool.acceptsDevicePixels();
    if (hostCopy == trt_alpha::datasource::HostCopy::On)
    {
        // 显式要求：即使没人读，也留着（源码里要拿 image() 就必须它）。
        srcCfg.needHostCopy = true;
    }
    else if (hostCopy == trt_alpha::datasource::HostCopy::Off)
    {
        // 显式要求不拷，但这两条中任一条成立时"不拷"根本不可能满足：渲染器会
        // 拿着显存指针当主机图像画，模型会在空缓冲上做 H2D。当场报错，绝不
        // 悄悄改回 true —— 那正是 ini 值静默失效。
        if (needRender)
        {
            throw std::runtime_error(
                "run: source.need_host_copy = off cannot be honoured together with "
                "--save / --show (the renderer draws into the host copy); "
                "use auto or on");
        }
        if (!pool.acceptsDevicePixels())
        {
            throw std::runtime_error(
                "run: source.need_host_copy = off cannot be honoured with model '" +
                opt.net + "' (it uploads pixels from the host); use auto or on");
        }
        if (decodeBackend == trt_alpha::datasource::DecodeBackend::OpenCV)
        {
            TRT_LOG_WARN("run: source.need_host_copy = off has no effect on the "
                         "OpenCV backend, which always decodes into host memory");
        }
        srcCfg.needHostCopy = false;
    }

    // 本次回拷决策的最终口径（与高层 Infer 的那条日志同名同序，便于对照）。
    TRT_LOG_INFO("run: host pixel copy " << (srcCfg.needHostCopy ? "ON" : "OFF")
                 << " (render=" << (needRender ? "on" : "off")
                 << ", model "
                 << (pool.acceptsDevicePixels() ? "supports" : "does not support")
                 << " device pixels, decode="
                 << trt_alpha::datasource::toString(srcCfg.decodeBackend)
                 << ", hwaccel=" << trt_alpha::datasource::toString(srcCfg.hwAccel)
                 << ", need_host_copy="
                 << trt_alpha::datasource::toString(hostCopy) << ")");

    std::vector<std::unique_ptr<trt_alpha::datasource::IDataSource>> sources;
    sources.push_back(trt_alpha::datasource::makeSource(srcCfg));

    trt_alpha::renderer::OpenCVRenderer renderer;

    // 把输入源登记给渲染器：save() 会拒绝对这些路径的写入（"输出 == 输入"的
    // 最后一道闸门，任何数据源都躲不过）。
    if (opt.save && !trt_alpha::core::Paths::isUrl(srcCfg.path))
    {
        renderer.addInputSource(trt_alpha::core::Paths::resolve(srcCfg.path).string());
    }

    // JPEG 质量（只影响 --save 落盘）：1..100，默认 95 = OpenCV 默认值。
    // 读取在函数前半段（见 jpegQuality），保证配置框不会把该键误标 [unused]。
    renderer.setJpegQuality(jpegQuality);

    // 队列满时策略：ini 的 pool.queue_full_policy 显式指定 > 按源类型自动。
    //   实时源（摄像头 / RTSP / HTTP 流）→ drop_oldest：宁可丢旧帧，也不让延迟累积
    //   离线源（单图 / 图片目录 / 视频文件）→ block：一帧不落（备忘录第 9 条的口径）
    trt_alpha::core::QueueFullPolicy queuePolicy = trt_alpha::core::QueueFullPolicy::Block;
    {
        const std::string pol = modelCfg.getString("pool.queue_full_policy", "auto");
        if (pol == "block")
        {
            queuePolicy = trt_alpha::core::QueueFullPolicy::Block;
        }
        else if (pol == "drop_oldest")
        {
            queuePolicy = trt_alpha::core::QueueFullPolicy::DropOldest;
        }
        else if (pol == "drop_newest")
        {
            queuePolicy = trt_alpha::core::QueueFullPolicy::DropNewest;
        }
        else
        {
            if (pol != "auto")
            {
                TRT_LOG_WARN("run: unknown pool.queue_full_policy '" << pol
                             << "', falling back to auto");
            }
            const bool realtime =
                (srcCfg.type == trt_alpha::datasource::SourceType::Camera) ||
                trt_alpha::core::Paths::isUrl(srcCfg.path);
            queuePolicy = realtime ? trt_alpha::core::QueueFullPolicy::DropOldest
                                   : trt_alpha::core::QueueFullPolicy::Block;
        }
    }

    // Pipeline
    trt_alpha::pipeline::PipelineConfig pcfg;
    pcfg.sources = std::move(sources);
    pcfg.pools = { &pool };
    pcfg.renderer = needRender ? &renderer : nullptr;
    pcfg.classNames = modelCfg.classNames;
    pcfg.renderOptions = renderOptions;
    pcfg.resultQueueSize = (qResult > 0) ? static_cast<std::size_t>(qResult) : std::size_t{32};
    pcfg.renderThreads = rThreads;
    pcfg.queueFullPolicy = queuePolicy;
    pcfg.saveEnabled = opt.save;
    pcfg.saveDir = saveDir;
    pcfg.showEnabled = opt.show;
    pcfg.showWindow = showWindow;

    trt_alpha::pipeline::Pipeline p(std::move(pcfg));
    p.start();

    if (p.hasRenderer())
    {
        // 渲染线程自己消费结果队列
        p.waitForCompletion();
    }
    else
    {
        // 没渲染器：结果不需要，但必须被消费 —— Block 策略下没人 pop，
        // 源线程写满队列后会永久阻塞。主线程在这里排空即可（零绘制开销）。
        trt_alpha::core::BatchResult discard;
        while (p.popResult(discard)) {}
        // 幂等收尾：确保队列关闭，任何阻塞在 push 上的源线程都能退出
        p.stop();
        p.waitForCompletion();
    }

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
