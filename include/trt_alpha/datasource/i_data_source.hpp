// =============================================================================
//  trt_alpha :: datasource :: i_data_source
// -----------------------------------------------------------------------------
//  IDataSource —— 数据源抽象接口。
//
//  职责：
//    * 从图片 / 视频 / 摄像头读帧
//    * 攒满一批（batchSize）产出 core::Batch
//    * 不满一批时：多余帧填 0，validCount 标记有效帧数
//
//  不负责：
//    * 推理 / 渲染 / 预处理
//    * 主动丢帧（用户自己管）
//
//  生命周期：
//    * 构造 = 打开资源（图片 / 视频 / 摄像头）
//    * next() = 读一批
//    * requestStop() = 请求停止（线程安全）
//    * 析构 = 释放资源
//
//  线程模型：
//    * 每个 IDataSource 实例由【一个数据源线程】调用
//    * requestStop() 可由其他线程调用（线程安全）
//
//  错误处理：
//    * 构造失败（文件不存在 / 摄像头打不开）→ 抛异常
//    * next() 读失败 → 返回 false（结束）或抛异常（意外错误）
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"

namespace trt_alpha::datasource {

class IDataSource
{
public:
    virtual ~IDataSource() = default;

    IDataSource(const IDataSource&) = delete;
    IDataSource& operator=(const IDataSource&) = delete;

    //! 读下一批。返回 false 表示"没有更多了"（文件读完 / 被请求停止）。
    //! 输出参数 out：
    //!   * out.buffer 是连续内存，大小 = batchSize × H × W × C
    //!   * out.views[i] 指向 buffer 的第 i 块
    //!   * out.validCount 标记有效帧数
    //!   * 不满的批：后 (batchSize - validCount) 帧填 0
    [[nodiscard]] virtual bool next(core::Batch& out) = 0;

    //! 请求停止（线程安全，可从任意线程调用）。
    //! 调用后，正在阻塞的 next() 应尽快返回 false。
    virtual void requestStop() = 0;

    //! 数据源类型名（日志 / 调试用）。
    [[nodiscard]] virtual const char* typeName() const noexcept = 0;

protected:
    IDataSource() = default;
};

}  // namespace trt_alpha::datasource