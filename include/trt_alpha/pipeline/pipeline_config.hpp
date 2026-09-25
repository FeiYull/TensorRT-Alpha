// =============================================================================
//  trt_alpha :: pipeline :: pipeline_config
// -----------------------------------------------------------------------------
//  PipelineConfig —— 三级流水线的配置。
//
//  数据流：
//    [DataSource] → [InferencePool] → [ResultQueue] → [Renderer]
//
//  多池 + 路由：
//    * pools[i] —— 第 i 个推理池（不同模型）
//    * sourceToPool[i] —— 源 i 用哪个池（下标）
//    * sourceToPool 为空时：所有源用 pools[0]
//
//  存盘 / 显示：
//    * saveEnabled：是否存盘
//    * saveDir：存盘目录（相对工程根或绝对路径）
//    * showEnabled：是否显示
//    * showWindow：显示窗口名
// =============================================================================
#pragma once

#include "trt_alpha/core/class_info.hpp"
#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/renderer/i_renderer.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::core {
class InferencePool;
}

namespace trt_alpha::pipeline {

struct PipelineConfig
{
    // ---- 数据源（用户创建，所有权转移给 Pipeline）----
    std::vector<std::unique_ptr<datasource::IDataSource>> sources;

    // ---- 推理池（用户创建，生命周期由用户管）----
    std::vector<core::InferencePool*> pools;

    // ---- 源 → 池 路由 ----
    //! 长度应等于 sources.size()。
    //! 空 = 所有源用 pools[0]。
    //! sourceToPool[i] = k 表示源 i 用 pools[k]。
    std::vector<std::size_t> sourceToPool;

    // ---- 渲染 ----
    renderer::IRenderer* renderer = nullptr;
    std::vector<core::ClassInfo> classNames;

    // ---- 结果队列 ----
    std::size_t resultQueueSize = 32;

    // ---- 存盘 / 显示 ----
    bool saveEnabled = false;
    std::string saveDir = "save";
    bool showEnabled = false;
    std::string showWindow = "trt_alpha";

    // ---- 源停止等待超时（ms） ----
    int stopTimeoutMs = 2000;
};

}  // namespace trt_alpha::pipeline