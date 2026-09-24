// =============================================================================
//  trt_alpha :: core :: engine
// -----------------------------------------------------------------------------
//  TrtEngine —— IRuntime / ICudaEngine / IExecutionContext 的 RAII 封装。
//
//  职责（只做引擎生命周期，不含任何预处理 / 后处理逻辑）：
//    * 从序列化 engine 文件（.trt / .engine）反序列化，失败抛异常
//    * 按名字枚举全部 I/O 张量（ioTensors()）
//    * 对动态 shape 引擎设置实际输入形状（setInputShape；静态引擎自动跳过）
//
//  TRT 10 专用 API（不兼容 8.x）：
//    getNbIOTensors / getIOTensorName / getTensorIOMode / getTensorShape
//    setInputShape / setTensorAddress / enqueueV3
//
//  生命周期：
//    * 构造 = 加载 engine + 创建 context；失败抛异常（RAII 回滚）
//    * 每个 IExecutionContext 只能被一个线程使用
//      （要并发 → 每线程构造一个 TrtEngine 实例，共享同一个 engine 文件）
//
//  TODO：
//    * buildFromOnnx 目前只声明 + 抛未实现；将来移到独立的 builder 模块
//      （因为 core 不依赖 nvonnxparser）
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

//! I/O 张量描述（名字 / 形状 / 元素类型 / 输入还是输出）。
//! shape 直接用 nvinfer1::Dims —— TrtEngine 就是 TRT 封装，绑死合理。
struct TensorDesc
{
    std::string name;
    nvinfer1::Dims shape{};
    DataType dtype = DataType::Float32;
    bool isInput = false;

    //! 元素个数（动态维 -1 视为 0；调用方应先 setInputShape 再取）。
    [[nodiscard]] std::size_t volume() const noexcept;
};

//! TensorRT 引擎的 RAII 封装。
class TrtEngine
{
public:
    //! ONNX → engine 转换的构建选项（TODO：将来移到 builder 模块）。
    struct BuildOptions
    {
        std::size_t workspaceBytes = 1ULL << 30;  //!< 1 GiB 构建 workspace
        bool fp16 = true;                         //!< 平台支持时开启
    };

    //! 从序列化 engine 文件加载。文件不存在 / 反序列化失败抛 std::runtime_error。
    explicit TrtEngine(const std::string& engineFile);

    //! ONNX → 序列化 engine（TODO：当前未实现，调用抛 std::logic_error）。
    //! 将来移到独立 builder 模块（core 不依赖 nvonnxparser）。
    static void buildFromOnnx(const std::string& onnxFile,
                              const std::string& engineFile,
                              const BuildOptions& options = {});

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    [[nodiscard]] nvinfer1::ICudaEngine* engine() noexcept { return m_engine.get(); }
    [[nodiscard]] nvinfer1::IExecutionContext* context() noexcept
    {
        return m_context.get();
    }

    //! 引擎全部 I/O 张量描述（构造时枚举一次）。
    [[nodiscard]] const std::vector<TensorDesc>& ioTensors() const noexcept
    {
        return m_io;
    }

    //! 按名字找张量描述；找不到返回 nullptr。
    [[nodiscard]] const TensorDesc* find(const std::string& name) const noexcept;

    //! 设置实际输入形状。
    //! 静态 shape 的引擎自动跳过（调 setInputShape 会让后续推理失败，
    //! 这是 TRT 10 与 8.x 行为差异点之一）。
    void setInputShape(const std::string& name, const nvinfer1::Dims& dims);

    //! context 级（已应用 setInputShape 后）的实际形状。
    [[nodiscard]] nvinfer1::Dims contextShape(const std::string& name) const;

private:
    void loadSerialized(const void* data, std::size_t size);
    void discoverIo();

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    std::vector<TensorDesc> m_io;
};

}  // namespace trt_alpha::core