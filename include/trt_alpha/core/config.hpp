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

//! 前置声明：logConfigBox 只按指针接收 I/O 描述，
//! 因此本头文件无需引入 engine.hpp（避免把 NvInfer.h 传染给所有使用者）。
struct TensorDesc;

//! INI → ModelConfig。
//! engine / classNamesFile / inputOutputNames / numClass 是必填；
//! 缺失时抛 std::runtime_error（消息带上下文）。
//! 相对路径原样保留（由调用方决定相对谁）。
//! 同时填写 cfg.origins：每个键来自 base.ini 还是本模型 ini（保序）。
[[nodiscard]] ModelConfig loadModelConfig(const std::string& iniPath);

//! TXT → vector<ClassInfo>。
//! 每行："名字 R G B"（RGB 0-255），行号 = label。
//! 格式错误 / 文件不可读 → 抛 std::runtime_error。
[[nodiscard]] std::vector<ClassInfo> loadClassNamesFile(const std::string& txtPath);

// -----------------------------------------------------------------------------
//  配置展示：把"本次实际生效的配置"整体框出来
// -----------------------------------------------------------------------------
//! 打印一个配置框（上下双横线、左右单竖线），Release 构建下同样输出。
//!   * 正文：base.ini + 模型 ini 合并后的【全部键】，按 section 分组；
//!     每行 = "短名 = 值 来源"。ini 里新增键自动出现，无需改代码。
//!   * 【真生效判定】：没被任何消费者读过的键会标 `[unused]`
//!     —— 由 ModelConfig::readKeys（getXxx 自动登记 + loadModelConfig 显式登记）判定。
//!   * 追加段 [resolved]：引擎真相（batch 是否被修正 / 每个输入输出的形状、
//!     dtype、物理格式）。io == nullptr 时不打该段。
//!
//! @param cfg           已加载、已应用 CLI 覆盖、且已被消费者读取过的配置。
//!                      建议传 InferencePool::modelConfig()（模型真正用那份，
//!                      readKeys 才完整）；传外层的 cfg 会把模型读的键误标 unused。
//! @param netName       模型注册名（标题行用）
//! @param iniPath       模型 ini 路径（标题行用）
//! @param io            引擎 I/O 张量描述（一般取 InferencePool::ioDesc()，
//!                      模型 init 之后才有效）；nullptr = 不打 [resolved] 段
//! @param resolvedBatch 引擎修正后的 batch；<=0 = 未知（不打该行）
void logConfigBox(const ModelConfig& cfg,
                  const std::string& netName,
                  const std::string& iniPath,
                  const std::vector<TensorDesc>* io = nullptr,
                  int resolvedBatch = 0);

}  // namespace trt_alpha::core