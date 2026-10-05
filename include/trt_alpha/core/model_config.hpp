// =============================================================================
//  trt_alpha :: core :: model_config
// -----------------------------------------------------------------------------
//  ModelConfig —— 模型运行配置。
//
//  设计：
//    * 通用字段（所有模型共用）：engine / batchSize / dstH / dstW /
//      inputOutputNames / classNamesFile / classNames
//    * 模型特有字段：统一放 extras（unordered_map<string, string>），
//      由模型自己用 getInt / getFloat / getString / getBool 读
//
//  为什么用 extras 而不是"每个模型一个 Config 子类"：
//    * IModel::init 签名保持 const ModelConfig&，无需 dynamic_cast
//    * 加新模型不用改 ModelConfig
//    * 配置本质就是"运行时字符串"，类型安全在读时保证
//
//  来源：
//    * INI 文件（configs/<model>.ini）
//    * CLI 参数覆盖部分字段
//    * classNames 运行时从类别文件读
// =============================================================================
#pragma once

#include "trt_alpha/core/class_info.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace trt_alpha::core {
    class Engine;   // 前置声明
}

namespace trt_alpha::core {

struct ModelConfig
{
    // ---- 所有模型共用的字段 ----
    std::string engine;                        //!< engine 路径（相对根 / 绝对）
    std::string classNamesFile;                //!< 类别文件路径（名字 + RGB）
    int batchSize = 1;                         //!< 本次推理实际使用的 batch（运行时按引擎能力修正）
    int maxBatchSize = -1;                     //!< 可选：声明的引擎 batch 上界契约；<=0 表示未声明
    int dstH = 640;
    int dstW = 640;
    std::vector<std::string> inputOutputNames;

    // ---- 模型特有字段（统一放这里）----
    // key = INI 原始 key（含 section 前缀，如 "model.num_class"）
    //     —— 同时也会存"去掉 section 的短名"（"num_class"）
    // value = INI 原始字符串值
    std::unordered_map<std::string, std::string> extras;

    // ---- 运行时填 ----
    std::vector<ClassInfo> classNames;

    //! 共享 engine（可选）。非空时，IModel::init 复用它，不再从 engine 路径反序列化。
    //! 用于"1 engine + N context"。
    std::shared_ptr<Engine> sharedEngine;

    // ---- 便捷读取（找不到返回 fallback；类型转换失败抛异常）----
    [[nodiscard]] std::string getString(const std::string& key,
                                       const std::string& fallback = "") const;
    [[nodiscard]] int getInt(const std::string& key, int fallback) const;
    [[nodiscard]] float getFloat(const std::string& key, float fallback) const;
    [[nodiscard]] bool getBool(const std::string& key, bool fallback) const;
};

}  // namespace trt_alpha::core