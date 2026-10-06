// =============================================================================
//  trt_alpha :: datasource :: source_config
// -----------------------------------------------------------------------------
//  SourceConfig —— 数据源配置。
// =============================================================================
#pragma once

#include <string>

namespace trt_alpha::datasource {

//! 数据源类型。
enum class SourceType
{
    Image,     //!< 单张图片
    Images,    //!< 图片目录（扫描常见格式）
    Video,     //!< 视频文件，或网络流 URL（rtsp / rtmp / http(s) / udp）
    Camera,    //!< USB / 网络摄像头
};

struct SourceConfig
{
    SourceType type = SourceType::Image;

    //! 图片 / 目录 / 视频路径（相对工程根或绝对路径），或流 URL。
    //! URL 由 core::Paths::isUrl() 识别：走 FFmpeg 后端直连，不做本地文件校验。
    //! type == Camera 时忽略。
    //! 注：URL 一律按流处理（Image 类型遇到 URL 会明确报错），因为
    //!     cv::imread 只认本地文件。
    std::string path;

    //! 摄像头 ID（type == Camera 时有效）。
    int cameraId = -1;

    //! 攒批大小（引擎 batch size）。
    int batchSize = 1;

    //! 视频是否循环播放（type == Video 时有效）。
    //! 仅对本地视频文件生效 —— 流无法回绕。
    bool loop = false;

    //! 流打开超时 / 读超时（毫秒，仅 FFmpeg/GStreamer 后端的 URL 流生效）。
    //! <= 0 = 交给后端默认（即无限等待，不可达的流会长时间阻塞）。
    //! 默认 5s：既给慢速流余量，又不让进程死等。
    int openTimeoutMs = 5000;
    int readTimeoutMs = 5000;

    //! 数据源标识（多源时用于分辨来源）。
    int sourceId = -1;
};

}  // namespace trt_alpha::datasource