// =============================================================================
//  trt_alpha :: core :: config
// -----------------------------------------------------------------------------
//  从 INI 文件加载 ModelConfig，以及从 TXT 加载类别信息。
// =============================================================================
#pragma once

#include "trt_alpha/core/class_info.hpp"
#include "trt_alpha/core/model_config.hpp"

#include <string>
#include <vector>

namespace trt_alpha::core {

//! INI → ModelConfig。
//! engine / classNamesFile / inputOutputNames / numClass 是必填；
//! 缺失时抛 std::runtime_error（消息带上下文）。
//! 相对路径原样保留（由调用方决定相对谁）。
[[nodiscard]] ModelConfig loadModelConfig(const std::string& iniPath);

//! TXT → vector<ClassInfo>。
//! 每行："名字 R G B"（RGB 0-255），行号 = label。
//! 格式错误 / 文件不可读 → 抛 std::runtime_error。
[[nodiscard]] std::vector<ClassInfo> loadClassNamesFile(const std::string& txtPath);

}  // namespace trt_alpha::core