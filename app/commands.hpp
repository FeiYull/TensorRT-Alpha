// =============================================================================
//  trt_alpha :: app :: commands
// =============================================================================
#pragma once

#include <string>
#include <vector>

namespace trt_alpha::app {

//! 打印用法。
void printUsage();

//! `trt_alpha list`：列出已注册模型。
int listCommand(const std::vector<std::string>& args);

//! `trt_alpha run`：图片/视频/摄像头推理。
int runCommand(const std::vector<std::string>& args);

//! `trt_alpha bench`：吞吐/延迟（TODO）。
int benchCommand(const std::vector<std::string>& args);

//! `trt_alpha build`：ONNX → engine（TODO）。
int buildCommand(const std::vector<std::string>& args);

}  // namespace trt_alpha::app