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

//! I/O 张量描述。
struct TensorDesc
{
    std::string name;
    nvinfer1::Dims shape{};
    DataType dtype = DataType::Float32;
    bool isInput = false;

    [[nodiscard]] std::size_t volume() const noexcept;
};

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

    //! 设置实际输入形状。静态 shape 的引擎自动跳过。
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