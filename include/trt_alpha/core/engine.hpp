// =============================================================================
//  trt_alpha :: core :: engine
// -----------------------------------------------------------------------------
//  Engine  —— 共享的 ICudaEngine 封装（线程安全，可被多个 Context 引用）。
//  Context —— 独占的 IExecutionContext 封装（非线程安全，每个 worker 一个）。
//  TrtEngine —— 兼容壳：持 shared_ptr<Engine> + 独占 Context。
//
//  1 engine + N context 模型：
//    * 反序列化 1 次 engine（权重 1 份）
//    * 每个 worker 1 个 context
//
//  兼容旧用法：
//    * TrtEngine(file) 会自己反序列化一份 engine（独立，不复用）
// =============================================================================
#pragma once

#include "trt_alpha/core/data_type.hpp"

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::core {

struct BatchRange
{
    int min = 1;
    int opt = 1;
    int max = 1;
    [[nodiscard]] bool isDynamic() const noexcept { return min != max; }
};

//! I/O 张量描述。
struct TensorDesc
{
    std::string name;
    nvinfer1::Dims shape{};      // 引擎声明形状（动态维为 -1）
    nvinfer1::Dims minShape{};   // profile kMIN
    nvinfer1::Dims optShape{};   // profile kOPT
    nvinfer1::Dims maxShape{};   // profile kMAX
    DataType dtype = DataType::Float32;
    bool isInput = false;

    [[nodiscard]] std::size_t volume() const noexcept;

    [[nodiscard]] bool isDynamicBatch() const noexcept
    { return shape.nbDims > 0 && shape.d[0] < 0; }

    [[nodiscard]] BatchRange batchRange() const noexcept;   // 见文件

private:
    [[nodiscard]] int pick(const nvinfer1::Dims& d, int fallback) const noexcept;
};

//! 引擎输入 batch 的解析结果（统一静态 / 动态语义）。
struct ResolvedBatch
{
    int  batch     = 1;      //!< 修正后实际使用的 batch
    bool isDynamic = false;  //!< 引擎输入是否为动态 batch
    bool corrected = false;  //!< 是否被框架修正过（与请求值不同）
    int  min = 1;            //!< 引擎 profile 最小 batch
    int  opt = 1;            //!< 引擎 profile 最优 batch
    int  max = 1;            //!< 引擎 profile 最大 batch
};

//! 依据引擎实际能力，解析 / 修正调用方请求的 batch：
//!   * 静态引擎（onnx 固定 batch）：batch 由引擎写死；
//!     请求值 != 引擎固定值 → WARN 并修正为引擎值。
//!   * 动态引擎（onnx -1 + trtexec min/opt/max）：clamp 到 [min, max]；
//!     请求值 > max → WARN 并修正为 max；
//!     请求值 < min → 抛 std::runtime_error（无法修正到合法值）。
//!   * 可选上界契约：declaredMax > 0 时，必须与引擎 profile max 一致，
//!     否则抛 std::runtime_error（配置声明的能力与引擎不符）。
//! @param input       引擎输入张量描述（需已填 profile 形状）
//! @param requested   调用方请求的 batch
//! @param who         调用方名字（用于日志与异常信息）
//! @param declaredMax 配置声明的上界契约；<=0 表示未声明
[[nodiscard]] ResolvedBatch resolveBatch(const TensorDesc& input,
                                         int requested,
                                         const std::string& who,
                                         int declaredMax = 0);

//! 共享的 ICudaEngine（线程安全，可被多个 Context 引用）。
class Engine
{
public:
    //! 从序列化 engine 文件加载。失败抛 std::runtime_error。
    explicit Engine(const std::string& engineFile);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    [[nodiscard]] nvinfer1::ICudaEngine* get() noexcept { return m_engine.get(); }
    [[nodiscard]] const nvinfer1::ICudaEngine* get() const noexcept { return m_engine.get(); }

    [[nodiscard]] const std::vector<TensorDesc>& ioTensors() const noexcept { return m_io; }
    [[nodiscard]] const TensorDesc* find(const std::string& name) const noexcept;

private:
    void discoverIo();

    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::vector<TensorDesc> m_io;
};

//! 独占的 IExecutionContext（非线程安全，每个 worker 一个）。
class Context
{
public:
    //! 从共享 engine 创建 context。engine 必须比 Context 活得更久。
    explicit Context(Engine& engine);

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    [[nodiscard]] nvinfer1::IExecutionContext* get() noexcept { return m_context.get(); }
    [[nodiscard]] const nvinfer1::IExecutionContext* get() const noexcept { return m_context.get(); }

    //! 设置实际输入形状。
    //!  * 动态张量：下发 TRT；越界（超出 profile）由 TRT 拒绝并抛 std::runtime_error。
    //!  * 静态张量：形状由引擎写死；与请求一致时静默跳过，
    //!    不一致时抛 std::runtime_error（配置 / 引擎不匹配，杜绝静默越界）。
    void setInputShape(const std::string& name, const nvinfer1::Dims& dims);

    //! context 级（已应用 setInputShape 后）的实际形状。
    [[nodiscard]] nvinfer1::Dims contextShape(const std::string& name) const;

private:
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    nvinfer1::ICudaEngine* m_engine = nullptr;   // 不拥有
};

//! 兼容壳：持 shared_ptr<Engine> + 独占 Context。
class TrtEngine
{
public:
    struct BuildOptions
    {
        std::size_t workspaceBytes = 1ULL << 30;
        bool fp16 = true;
    };

    //! 旧构造：自己反序列化一份 engine（独立，不复用）。
    explicit TrtEngine(const std::string& engineFile);

    //! 新构造：复用共享 engine（1 engine + N context）。
    explicit TrtEngine(std::shared_ptr<Engine> sharedEngine);

    //! ONNX → engine（TODO）。
    static void buildFromOnnx(const std::string& onnxFile,
                              const std::string& engineFile,
                              const BuildOptions& options = {});

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    [[nodiscard]] nvinfer1::ICudaEngine* engine() noexcept
    {
        return m_sharedEngine ? m_sharedEngine->get() : nullptr;
    }
    [[nodiscard]] nvinfer1::IExecutionContext* context() noexcept
    {
        return m_context ? m_context->get() : nullptr;
    }
    [[nodiscard]] const std::vector<TensorDesc>& ioTensors() const noexcept
    {
        static const std::vector<TensorDesc> kEmpty;
        return m_sharedEngine ? m_sharedEngine->ioTensors() : kEmpty;
    }
    [[nodiscard]] const TensorDesc* find(const std::string& name) const noexcept
    {
        return m_sharedEngine ? m_sharedEngine->find(name) : nullptr;
    }
    [[nodiscard]] const Engine* sharedEngine() const noexcept
    {
        return m_sharedEngine.get();
    }

    void setInputShape(const std::string& name, const nvinfer1::Dims& dims)
    {
        if (m_context) { m_context->setInputShape(name, dims); }
    }
    [[nodiscard]] nvinfer1::Dims contextShape(const std::string& name) const
    {
        return m_context ? m_context->contextShape(name) : nvinfer1::Dims{};
    }

private:
    std::shared_ptr<Engine> m_sharedEngine;
    std::unique_ptr<Context> m_context;
};

}  // namespace trt_alpha::core