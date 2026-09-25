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
    Video,     //!< 视频文件
    Camera,    //!< USB / 网络摄像头
};

struct SourceConfig
{
    SourceType type = SourceType::Image;

    //! 图片 / 目录 / 视频路径（相对工程根或绝对路径）。
    //! type == Camera 时忽略。
    std::string path;

    //! 摄像头 ID（type == Camera 时有效）。
    int cameraId = -1;

    //! 攒批大小（引擎 batch size）。
    int batchSize = 1;

    //! 视频是否循环播放（type == Video 时有效）。
    bool loop = false;

    //! 数据源标识（多源时用于分辨来源）。
    int sourceId = -1;
};

}  // namespace trt_alpha::datasource