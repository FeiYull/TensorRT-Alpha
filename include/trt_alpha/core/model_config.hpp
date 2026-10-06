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
#include "trt_alpha/core/layout.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
    //! 输入【逻辑维序】（INI: input.layout，如 nchw / nhwc / ncdhw）。
    //! 空 = 用模型自身的规范布局。H/W 不再需要配置：以引擎声明形状为唯一真相源。
    Layout layout;
    //! 空间维"意图值"（INI: input.dst_h / input.dst_w）。0 = 未设置（默认）。
    //! 一般不需要设置：静态维一律以引擎声明形状为准；
    //! 仅当引擎该维是动态维（-1）时，这里才作为"跑多大"的意图值生效。
    int dstH = 0;
    int dstW = 0;
    std::vector<std::string> inputOutputNames;

    // ---- 模型特有字段（统一放这里）----
    // key = INI 原始 key（含 section 前缀，如 "model.num_class"）
    //     —— 同时也会存"去掉 section 的短名"（"num_class"）
    // value = INI 原始字符串值
    std::unordered_map<std::string, std::string> extras;

    //! 每个键（只记"长名"，如 "model.engine"）的【来源】+【出现顺序】。
    //! source 取值："base.ini" / "<模型>.ini" / "CLI"。
    //! 顺序 = base.ini 出现顺序，模型 ini 新增的键追加在后（重复键保持首次位置）。
    //! 仅供配置展示（logConfigBox）使用，不参与任何逻辑判定。
    std::vector<std::pair<std::string, std::string>> origins;

    //! 写入 / 改写某个长名键的来源（CLI 覆盖配置时调用）。保持首次出现顺序。
    void setOrigin(const std::string& fullKey, const std::string& source);

    //! 查询某个长名键的来源；找不到返回 fallback。
    [[nodiscard]] std::string originOf(const std::string& fullKey,
                                       const std::string& fallback = "-") const;

    //! 记录"被真正读取过"的键（短名 / 长名都可能，读到什么记什么）。
    //!   * 由 getInt / getFloat / getString / getBool 自动登记；
    //!   * loadModelConfig 对"直接走 IniParser"的那批键显式登记。
    //! 用途：logConfigBox 借此标出"ini 里写了、本次路径却没读"的死键
    //!（典型：键写错 section → 静默失效）。纯诊断，不参与任何逻辑判定。
    //! 注意：读取发生在 init 阶段（每个模型单线程），之后只读，无需加锁。
    mutable std::unordered_set<std::string> readKeys;

    //! 登记一个键已被消费（短名 / 长名均可）。
    void markRead(const std::string& key) const { readKeys.insert(key); }

    //! 长名键是否被消费过：精确命中，或去掉 section 后的短名命中。
    [[nodiscard]] bool wasRead(const std::string& fullKey) const;

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