// =============================================================================
//  trt_alpha :: datasource :: opencv_source（实现）
// =============================================================================
#include "trt_alpha/datasource/opencv_source.hpp"

#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/paths.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp> 

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace trt_alpha::datasource {
namespace {

//! 判断是否为常见图片扩展名（小写比较）。
bool isImageFile(const fs::path& p)
{
    static const std::vector<std::string> kExts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
    };
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return std::find(kExts.begin(), kExts.end(), ext) != kExts.end();
}

//! 从流 URL 里取末段路径当存盘名主干（去查询串 / 扩展名 / 非法文件名字符）。
//! 例：rtsp://cam/live/main -> "main"；http://x/v/a.mp4?t=1 -> "a"；取不到 -> "stream"。
std::string urlStem(const std::string& url)
{
    std::string s = url.substr(0, url.find_first_of("?#"));
    while (!s.empty() && s.back() == '/')
    {
        s.pop_back();
    }
    const std::size_t slash = s.find_last_of('/');
    std::string name = (slash == std::string::npos) ? s : s.substr(slash + 1);

    const std::size_t dot = name.find_last_of('.');
    if (dot != std::string::npos && dot != 0)
    {
        name.resize(dot);
    }
    // 纯点（"." / ".."）无意义，且会生成隐藏文件名
    if (name.find_first_not_of('.') == std::string::npos)
    {
        name.clear();
    }

    // 文件名不允许 : * ? " < > | / \ —— 非 [0-9A-Za-z._-] 一律换成 '_'
    for (char& c : name)
    {
        const unsigned char u = static_cast<unsigned char>(c);
        if (!(std::isalnum(u) || c == '.' || c == '_' || c == '-'))
        {
            c = '_';
        }
    }
    return name.empty() ? std::string("stream") : name;
}

}  // namespace

OpenCVSource::OpenCVSource(const SourceConfig& cfg)
    : m_cfg(cfg)
{
    // 静默 OpenCV 的 plugin / backend 探测日志
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    if (m_cfg.batchSize <= 0)
    {
        throw std::runtime_error("OpenCVSource: batchSize must be > 0");
    }

    switch (m_cfg.type)
    {
    case SourceType::Image:  openImage();  break;
    case SourceType::Images: openImages(); break;
    case SourceType::Video:  openVideo();  break;
    case SourceType::Camera: openCamera(); break;
    }

    TRT_LOG_INFO("OpenCVSource[" << typeName() << "]: opened '"
                 << m_cfg.path << "' batch=" << m_cfg.batchSize
                 << " sourceId=" << m_cfg.sourceId);
}

OpenCVSource::~OpenCVSource()
{
    if (m_capture.isOpened())
    {
        m_capture.release();
    }
}

void OpenCVSource::openImage()
{
    // URL 是流，不是本地图片（cv::imread 只认本地文件）。
    // 明确报错，避免用户拿 --image 传流后收到误导的 "image not found"。
    if (core::Paths::isUrl(m_cfg.path))
    {
        throw std::runtime_error(
            "OpenCVSource: '" + m_cfg.path +
            "' is a URL (stream), not an image; use the video/stream source "
            "instead (CLI: --video <url>, InferParams.source = <url>)");
    }

    const fs::path p = core::Paths::resolve(m_cfg.path);

    // 传目录 = 目录批量。判定放在这里（而不是各调用方）：
    // Infer 的 source / CLI 的 --image / sample 传目录都能自动生效。
    // 口径：只扫一层，不递归子目录；目录里没有图片由 openImages() 抛错。
    std::error_code ec;
    if (fs::is_directory(p, ec))
    {
        m_cfg.type = SourceType::Images;   // 让 typeName() / 日志反映真实模式
        TRT_LOG_INFO("OpenCVSource: '" << core::Paths::toDisplay(p)
                     << "' is a directory -> images mode (single level)");
        openImages();
        return;
    }

    const fs::path file = core::Paths::requireFile(m_cfg.path, "image");
    m_imagePaths = { file.string() };
    m_imageIndex = 0;
}

void OpenCVSource::openImages()
{
    const fs::path dir = core::Paths::resolve(m_cfg.path);
    if (!fs::is_directory(dir))
    {
        throw std::runtime_error("OpenCVSource: not a directory: " +
                                 core::Paths::toDisplay(dir));
    }

    m_imagePaths.clear();
    for (const auto& entry : fs::directory_iterator(dir))
    {
        if (entry.is_regular_file() && isImageFile(entry.path()))
        {
            m_imagePaths.push_back(entry.path().string());
        }
    }
    if (m_imagePaths.empty())
    {
        throw std::runtime_error("OpenCVSource: no image files in " +
                                 core::Paths::toDisplay(dir));
    }
    std::sort(m_imagePaths.begin(), m_imagePaths.end());
    m_imageIndex = 0;

    TRT_LOG_INFO("OpenCVSource: found " << m_imagePaths.size()
                 << " image(s) in " << core::Paths::toDisplay(dir));
}

void OpenCVSource::openVideo()
{
    // 网络流（rtsp / rtmp / http(s) / udp ...）：没有本地文件可校验，
    // 跳过 requireFile，交给 FFmpeg 后端直连。
    // 显式指定 CAP_FFMPEG：Windows 上 MSMF 会抢先接管 http(s)，且不支持 rtsp。
    if (core::Paths::isUrl(m_cfg.path))
    {
        m_isStream = true;

        // 打开 / 读取超时（毫秒，仅 FFmpeg/GStreamer 后端支持）：
        // 没有它，不可达的流会按后端默认值长时间阻塞（实测 rtsp 默认约 30s）。
        std::vector<int> params;
        if (m_cfg.openTimeoutMs > 0)
        {
            params.push_back(cv::CAP_PROP_OPEN_TIMEOUT_MSEC);
            params.push_back(m_cfg.openTimeoutMs);
        }
        if (m_cfg.readTimeoutMs > 0)
        {
            params.push_back(cv::CAP_PROP_READ_TIMEOUT_MSEC);
            params.push_back(m_cfg.readTimeoutMs);
        }

        if (!m_capture.open(m_cfg.path, cv::CAP_FFMPEG, params))
        {
            throw std::runtime_error(
                "OpenCVSource: cannot open stream: " + m_cfg.path +
                "\n  hint: check the URL is reachable and the scheme is one of "
                "rtsp / rtmp / http / https / udp");
        }
        TRT_LOG_INFO("OpenCVSource: opened stream " << m_cfg.path
                     << " (openTimeout=" << m_cfg.openTimeoutMs
                     << "ms readTimeout=" << m_cfg.readTimeoutMs << "ms)");
        return;
    }

    const fs::path p = core::Paths::requireFile(m_cfg.path, "video");

    // 显式钉住 FFmpeg：Windows 上 CAP_ANY 可能绑到 MSMF 插件，同一个 1080p 文件
    // 实测解码 7.1 ms/帧 vs FFmpeg 3.0 ms/帧（差 2.4x）。URL 分支上面已经钉过，
    // 这里是为了本地文件也行为一致、不随机器/构建漂移。
    if (!m_capture.open(p.string(), cv::CAP_FFMPEG))
    {
        TRT_LOG_WARN("OpenCVSource: CAP_FFMPEG could not open '"
                     << core::Paths::toDisplay(p)
                     << "', retrying with the backend default");
        if (!m_capture.open(p.string()))
        {
            throw std::runtime_error("OpenCVSource: cannot open video: " +
                                     core::Paths::toDisplay(p));
        }
    }
}

void OpenCVSource::openCamera()
{
    if (m_cfg.cameraId < 0)
    {
        throw std::runtime_error("OpenCVSource: cameraId must be >= 0");
    }
    if (!m_capture.open(m_cfg.cameraId))
    {
        throw std::runtime_error("OpenCVSource: cannot open camera " +
                                 std::to_string(m_cfg.cameraId));
    }
}

const char* OpenCVSource::typeName() const noexcept
{
    switch (m_cfg.type)
    {
    case SourceType::Image:  return "image";
    case SourceType::Images: return "images";
    case SourceType::Video:  return "video";
    case SourceType::Camera: return "camera";
    }
    return "unknown";
}

void OpenCVSource::requestStop()
{
    m_stopRequested.store(true);
}

void OpenCVSource::copyIntoBatch(core::Batch& batch, int index, const cv::Mat& img)
{
    const core::BufferView& v = batch.views[static_cast<std::size_t>(index)];

    // 批内尺寸必须一致：批 buffer 与 view 的 stride/height 都按【本批首帧】定死，
    // 混入不同分辨率的帧会写到槽位外（越界），且后处理几何全错。
    // 分辨率一致性是数据源提供方的责任，这里只做守卫：不一致立即终止，绝不静默带病推理。
    if (img.cols != v.width || img.rows != v.height)
    {
        throw std::runtime_error(
            "OpenCVSource: frame " + std::to_string(index) +
            " is " + std::to_string(img.cols) + "x" + std::to_string(img.rows) +
            " but this batch is " + std::to_string(v.width) + "x" +
            std::to_string(v.height) + " (from frame 0); "
            "frames within one batch must share the same resolution - "
            "fix the source (image dir / stream) instead of mixing sizes");
    }

    const std::size_t oneFrame =
        static_cast<std::size_t>(v.stride) * static_cast<std::size_t>(v.height);
    std::uint8_t* dst = batch.buffer->mutableData() +
                        static_cast<std::size_t>(index) * oneFrame;

    const int rowBytes = img.cols * 3;

    // 连续 + 紧凑：一句话拷贝；否则逐行
    if (img.isContinuous() && v.stride == rowBytes)
    {
        std::memcpy(dst, img.data,
                    static_cast<std::size_t>(rowBytes) * img.rows);
    }
    else
    {
        for (int y = 0; y < img.rows; ++y)
        {
            std::memcpy(dst + static_cast<std::size_t>(y) * v.stride,
                        img.ptr(y), rowBytes);
        }
    }
}

std::string OpenCVSource::nameForNextFrame(std::uint64_t frameIndex) const
{
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "%06llu",
                  static_cast<unsigned long long>(frameIndex));

    switch (m_cfg.type)
    {
    case SourceType::Image:
    case SourceType::Images:
        // 图片：原文件名主干（bus.jpg -> "bus"），存盘时按原名落盘
        if (m_imageIndex < m_imagePaths.size())
        {
            return fs::path(m_imagePaths[m_imageIndex]).stem().string();
        }
        return {};

    case SourceType::Video:
        // 视频：源文件主干 + 帧号（同批多帧互不覆盖）；
        // 流（URL）没有本地文件名，从 URL 末段取主干。
        return (m_isStream ? urlStem(m_cfg.path)
                           : fs::path(m_cfg.path).stem().string()) + "_" + suffix;

    case SourceType::Camera:
        return "cam" + std::to_string(m_cfg.cameraId) + "_" + suffix;
    }
    return {};
}

bool OpenCVSource::readOneImage(cv::Mat& out)
{
    if (m_imageIndex >= m_imagePaths.size())
    {
        return false;
    }
    const std::string& path = m_imagePaths[m_imageIndex];
    out = cv::imread(path, cv::IMREAD_COLOR);
    ++m_imageIndex;

    // 文件在磁盘上存在（requireFile / 目录枚举都过了），却解不出像素 ——
    // 说明它根本不是图片 / 已损坏 / 编码不受支持（典型：把 .txt 当 --source 传进来）。
    // 这里绝不能像"源已耗尽"那样返回 false：那会让 Pipeline 正常收尾、run 以
    // exit 0 静默结束，用户以为跑成功了。显式抛错 -> sourceLoop 捕获 -> 退出码 1。
    if (out.empty())
    {
        throw std::runtime_error(
            "OpenCVSource: cannot decode image '" +
            core::Paths::toDisplay(fs::path(path)) +
            "' (not an image / corrupted / unsupported codec)");
    }
    return true;
}

bool OpenCVSource::readOneFrame(cv::Mat& out)
{
    if (!m_capture.read(out))
    {
        // 循环仅对本地视频文件有效：流不能回绕（CAP_PROP_POS_FRAMES 对流无效）。
        if (m_cfg.loop && m_cfg.type == SourceType::Video && !m_isStream)
        {
            // 循环：回到开头
            m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
            return m_capture.read(out);
        }
        return false;
    }
    return !out.empty();
}

bool OpenCVSource::next(core::Batch& out)
{
    if (m_stopRequested.load())
    {
        return false;
    }

    // 读第一帧（决定尺寸）—— 名字要在读之前取（readOneImage 会推进索引）
    const std::string firstName = nameForNextFrame(m_nextFrameIndex);
    cv::Mat first;
    bool gotFirst = false;
    if (m_cfg.type == SourceType::Image || m_cfg.type == SourceType::Images)
    {
        gotFirst = readOneImage(first);
    }
    else
    {
        gotFirst = readOneFrame(first);
    }
    if (!gotFirst)
    {
        return false;
    }

    const int W = first.cols;
    const int H = first.rows;


    // 每批新建 buffer（不复用）—— 避免"下一批覆盖上一批"
    out.sourceId = m_cfg.sourceId;
    out.firstFrameIndex = m_nextFrameIndex;
    out.buffer = core::Buffer::createHost(W * m_cfg.batchSize, H, 3,
                                          core::DataType::UInt8);
    out.views.clear();
    out.validCount = 0;
    out.frameNames.clear();

    const std::size_t oneFrame = static_cast<std::size_t>(W) * H * 3;
    for (int i = 0; i < m_cfg.batchSize; ++i)
    {
        core::BufferView v;
        v.data = out.buffer->data() + static_cast<std::size_t>(i) * oneFrame;
        v.width = W;
        v.height = H;
        v.stride = W * 3;
        v.channels = 3;
        v.dtype = core::DataType::UInt8;
        v.space = core::MemorySpace::Host;
        out.views.push_back(v);
    }

    // 填第一帧
    copyIntoBatch(out, 0, first);
    out.validCount = 1;
    out.frameNames.push_back(firstName);

    // 本批最多填几帧 —— 图片源恒为 1：
    //   * 图片目录里各图尺寸通常不同，而一个批 buffer 的 stride / height 是按
    //     本批首帧定死的，"批内必须同尺寸"是硬约束。把两张不同尺寸的图塞进同一
    //     批会撞 copyIntoBatch 的守卫（--images data/ 就是这么炸的）。
    //   * 一次推理只处理一张图，语义上也更清楚。
    //   注意：views 仍然铺满 m_cfg.batchSize 个槽、buffer 尺寸也不变 —— 静态
    //   batch 的引擎必须喂满 N 个槽，少给就是让引擎越界读。没填的槽补 0，
    //   并用 validCount 标出有效张数；后处理 / 渲染都按 validCount 跳过补零槽。
    //   视频 / 摄像头同源同尺寸，按 batchSize 攒帧（攒批在这里才有意义）。
    const int wantFrames = (m_cfg.type == SourceType::Image ||
                            m_cfg.type == SourceType::Images)
                               ? 1
                               : m_cfg.batchSize;

    // 继续填剩余帧
    for (int i = 1; i < wantFrames; ++i)
    {
        if (m_stopRequested.load())
        {
            break;
        }
        const std::string name = nameForNextFrame(m_nextFrameIndex + static_cast<std::uint64_t>(i));
        cv::Mat img;
        bool ok = false;
        if (m_cfg.type == SourceType::Image || m_cfg.type == SourceType::Images)
        {
            ok = readOneImage(img);
        }
        else
        {
            ok = readOneFrame(img);
        }
        if (!ok)
        {
            break;
        }
        copyIntoBatch(out, i, img);
        out.validCount += 1;
        out.frameNames.push_back(name);
    }

    // 不满的帧填 0（垃圾）
    if (out.validCount < m_cfg.batchSize)
    {
        std::uint8_t* dst = out.buffer->mutableData() +
                            static_cast<std::size_t>(out.validCount) * oneFrame;
        const std::size_t remain =
            static_cast<std::size_t>(m_cfg.batchSize - out.validCount) * oneFrame;
        std::memset(dst, 0, remain);
    }

    // 只按【实际消费掉的帧数】推进绝对帧号。之前固定 += batchSize，在末批
    // 不足 / 图片源每批 1 张时会把帧号跳空（firstFrameIndex 是渲染器回退命名的
    // 依据，跳空会让 fallback 名字出现空洞）。
    m_nextFrameIndex += static_cast<std::uint64_t>(out.validCount);

    TRT_LOG_DEBUG("OpenCVSource[" << typeName() << "]: batch @frame "
                  << out.firstFrameIndex
                  << " validCount=" << out.validCount << "/" << m_cfg.batchSize);

    return out.validCount > 0;
}

}  // namespace trt_alpha::datasource
