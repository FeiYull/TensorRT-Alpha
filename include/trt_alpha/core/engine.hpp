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
#include "trt_alpha/core/layout.hpp"
#include "trt_alpha/core/model_config.hpp"

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
    //! 物理内存排布（线性 / 分块向量化）。与 shape 的【逻辑次序】正交：
    //! shape 决定哪根轴是 H/W/C，format 决定这些值在内存里怎么摆。
    nvinfer1::TensorFormat format = nvinfer1::TensorFormat::kLINEAR;
    std::string formatDesc;      //!< 人类可读的格式名（日志 / 异常信息用）
    bool isInput = false;

    [[nodiscard]] std::size_t volume() const noexcept;

    //! N 轴（batch 轴）在布局里的下标；布局未声明 batch 轴时返回 -1。
    //! 有了它，"哪根轴是 batch" 才由【布局】决定，而不是硬写轴 0 ——
    //! CHW（无 N 轴）/ HWCN 这类布局才不会读错轴。
    [[nodiscard]] int batchAxisIndex(const Layout& layout) const noexcept
    { return layout.has(Layout::kBatch) ? layout.indexOf(Layout::kBatch) : -1; }

    //! 指定轴是否为动态维（-1）。axis < 0（无 batch 轴）恒为 false。
    [[nodiscard]] bool isDynamicBatch(int axis = 0) const noexcept
    { return axis >= 0 && axis < shape.nbDims && shape.d[axis] < 0; }

    //! 指定轴的 batch 区间（读 profile 的 min/opt/max）。axis < 0 返回 {1,1,1}。
    [[nodiscard]] BatchRange batchRange(int axis = 0) const noexcept;   // 见文件

private:
    [[nodiscard]] int pick(const nvinfer1::Dims& d, int axis, int fallback) const noexcept;
};

//! 引擎输入 batch 的解析结果（统一静态 / 动态语义）。
struct ResolvedBatch
{
    int  batch     = 1;      //!< 实际使用的 batch（== 请求值；不符时不会返回，直接抛异常）
    bool isDynamic = false;  //!< 引擎输入是否为动态 batch
    int  min = 1;            //!< 引擎 profile 最小 batch
    int  opt = 1;            //!< 引擎 profile 最优 batch
    int  max = 1;            //!< 引擎 profile 最大 batch
};

//! 依据引擎实际能力，校验调用方请求的 batch —— 不符一律抛 std::runtime_error：
//!   * 静态引擎（onnx 固定 batch）：batch 由引擎写死，请求值必须等于该固定值；
//!     不等 → 抛（**不静默纠正** —— 纠正会让人以为 ini / CLI 里的值生效了）。
//!   * 动态引擎（onnx -1 + trtexec min/opt/max）：batch 必须落在 [min, max] 内；
//!     请求值 > max 或 < min → 抛（**不静默钳制** —— 配置超出引擎能力是错误，
//!     静默降级会让人以为设置生效，且数据源可能仍按原值打包）。
//!   * 可选上界契约：declaredMax > 0 时，必须与引擎 profile max 一致，
//!     否则抛 std::runtime_error（配置声明的能力与引擎不符）。
//! 一句话：配置只表达"意图"，引擎 profile 是唯一真相源；冲突时显式失败，绝不静默改值。
//! @param input       引擎输入张量描述（需已填 profile 形状）
//! @param requested   调用方请求的 batch
//! @param who         调用方名字（用于日志与异常信息）
//! @param declaredMax 配置声明的上界契约；<=0 表示未声明
//! @param batchAxis   batch 轴下标（由布局决定，见 TensorDesc::batchAxisIndex）；
//!                    -1 表示该布局没有 batch 轴（如 CHW），此时 batch 概念上恒为 1
[[nodiscard]] ResolvedBatch resolveBatch(const TensorDesc& input,
                                         int requested,
                                         const std::string& who,
                                         int declaredMax = 0,
                                         int batchAxis = 0);

//! 引擎输入【空间维】的解析结果（按语义命名，与布局的排列无关）。
struct ResolvedInputShape
{
    int  depth     = 0;      //!< 布局无 D 轴 / 引擎未声明时为 0
    int  height    = 0;
    int  width     = 0;
    bool dynamic   = false;  //!< 空间维中存在动态（-1）轴（此时采用调用方意图值）
    bool corrected = false;  //!< 静态维被引擎纠正过（与意图值不同）
};

//! 校验输入张量的【秩 / 通道轴 / 物理格式】，任一不符抛 std::runtime_error：
//!   * shape.nbDims != layout.rank()  → 布局声明与引擎不符
//!   * C 轴静态尺寸 != channels       → 布局声明写错（防止把 H/W 当通道、静默错读）
//!   * format != kLINEAR             → 引擎要求分块/向量化排布，本框架只喂线性 buffer
//! 这三条是"安全不放步"的护栏：宁可明确报错，也不静默拿错轴。
//! @param channels 期望通道数；<=0 表示跳过通道校验
void validateInputTensor(const TensorDesc& input, const Layout& layout,
                         int channels, const std::string& who);

//! 以【引擎声明形状】为唯一真相源解析输入的空间维（D/H/W）。
//!   * 静态轴（> 0）：忽略 intent，采用引擎值；与 intent 不同则 corrected = true
//!   * 动态轴（-1）  ：采用 intent（调用方意图值）；intent 未提供则为 0
//! 前置校验同 validateInputTensor。
void resolveInputShape(const TensorDesc& input, const Layout& layout, int channels,
                       const ResolvedInputShape& intent, const std::string& who,
                       ResolvedInputShape& out);

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

//! 一步到位：落定输入 batch、解析输入布局 / 空间维并下发 setInputShape，
//! 结果就地写回 cfg.batchSize / cfg.dstH / cfg.dstW。
//!   * batch：先过 resolveBatch（引擎 profile 为唯一真相源，不符即抛）
//!   * 布局优先级：cfg.layout（INI 的 input.layout）> modelLayout（模型规范布局）
//!   * 目标形状按 layout 逐轴构造 → 天然支持 3~8 维的任意排列（NCHW/NHWC/NCDHW/CHWN/…）
//!   * 配置里的 dst_h / dst_w 降级为"意图值"：仅在引擎对应维为动态（-1）时生效；
//!     静态维一律以引擎为准并 WARN 纠正。
//! 失败（引擎无此输入 / 校验不过 / 某轴尺寸无法确定）抛 std::runtime_error。
void applyInputShape(TrtEngine& engine, const std::string& tensorName,
                     const Layout& modelLayout, int channels, ModelConfig& cfg);

}  // namespace trt_alpha::core