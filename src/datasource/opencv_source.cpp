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
    const fs::path p = core::Paths::requireFile(m_cfg.path, "image");
    m_imagePaths = { p.string() };
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
    const fs::path p = core::Paths::requireFile(m_cfg.path, "video");
    if (!m_capture.open(p.string()))
    {
        throw std::runtime_error("OpenCVSource: cannot open video: " +
                                 core::Paths::toDisplay(p));
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
    // batch.buffer 的布局：连续内存 [batchSize 张图]
    const std::size_t oneFrame = static_cast<std::size_t>(img.cols) * img.rows * 3;
    std::uint8_t* dst = batch.buffer->mutableData() + static_cast<std::size_t>(index) * oneFrame;

    // 逐行拷贝（cv::Mat 可能有 padding）
    const int rowBytes = img.cols * 3;
    for (int y = 0; y < img.rows; ++y)
    {
        std::memcpy(dst + static_cast<std::size_t>(y) * rowBytes,
                    img.ptr(y), rowBytes);
    }
}

bool OpenCVSource::readOneImage(cv::Mat& out)
{
    if (m_imageIndex >= m_imagePaths.size())
    {
        return false;
    }
    out = cv::imread(m_imagePaths[m_imageIndex], cv::IMREAD_COLOR);
    ++m_imageIndex;
    return !out.empty();
}

bool OpenCVSource::readOneFrame(cv::Mat& out)
{
    if (!m_capture.read(out))
    {
        if (m_cfg.loop && m_cfg.type == SourceType::Video)
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

    // 读第一帧（决定尺寸）
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

    // 继续填剩余帧
    for (int i = 1; i < m_cfg.batchSize; ++i)
    {
        if (m_stopRequested.load())
        {
            break;
        }
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
        ++out.validCount;
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

    m_nextFrameIndex += static_cast<std::uint64_t>(m_cfg.batchSize);

    TRT_LOG_DEBUG("OpenCVSource[" << typeName() << "]: batch #"
                  << (m_nextFrameIndex / m_cfg.batchSize - 1)
                  << " validCount=" << out.validCount << "/" << m_cfg.batchSize);

    return out.validCount > 0;
}

}  // namespace trt_alpha::datasource