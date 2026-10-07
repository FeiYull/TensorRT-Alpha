// =============================================================================
//  trt_alpha :: core :: model
// -----------------------------------------------------------------------------
//  IModel —— 所有任务（检测 / 分割 / 分类 / 未来任务）的统一总接口。
//
//  设计目标：
//    1. 任务无关：IModel 只约定"一批图像进来、结果在哪里取"，不约定输出
//       内存布局。具体输出结构体由任务基类（det::IDetector / seg::ISegmentor
//       / cls::IClassifier）定义
//    2. 按 batch 设计：setBatch 接收一批同尺寸图像（Batch）
//    3. 加新模型 = 1 个 .cpp + 1 行 TRT_ALPHA_REGISTER_MODEL，无侵入
//
//  调用时序契约（唯一合法顺序）：
//    init(cfg) -> [ setBatch(batch) -> preprocess() -> infer() -> postprocess()
//                   -> commitResult(result) -> reset() ] 循环
//    postprocess() 返回即本轮 GPU 结果就绪（内部 D2H + 流同步）。
//
//  错误处理契约：
//    初始化 / 参数 / 设备错误一律抛 std::runtime_error（带上下文）。
//    init() 内部应校验 ModelConfig（如 numClass 与引擎实际 nc 是否一致）。
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_config.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace trt_alpha {

//! 所有任务的统一总接口。
class IModel
{
public:
    virtual ~IModel() = default;

    IModel(const IModel&) = delete;
    IModel& operator=(const IModel&) = delete;

    //! 注册名（与 INI 里的 model 字段一致）。
    [[nodiscard]] virtual const std::string& name() const noexcept = 0;

    //! 初始化：加载 engine、分配显存。失败抛异常。
    //! 内部应校验 ModelConfig（如 numClass 与引擎实际 nc）。
    virtual void init(const core::ModelConfig& cfg) = 0;

    //! 一批图像上载显存。Batch 里 buffer 是连续内存。
    virtual void setBatch(const core::Batch& batch) = 0;

    //! CUDA 预处理（letterbox / 归一化 / HWC→NCHW 等）。模型自己实现。
    virtual void preprocess() = 0;

    //! enqueueV3 异步推理。
    virtual void infer() = 0;

    //! 解码 + NMS + D2H；返回即结果就绪。模型自己实现。
    virtual void postprocess() = 0;

    //! 把本轮结果 move 到 out（commit 语义）。
    virtual void commitResult(core::BatchResult& out) = 0;

    //! 清空本轮状态。
    virtual void reset() = 0;

    //! 引擎 I/O 张量描述（调试用）。
    [[nodiscard]] virtual const std::vector<core::TensorDesc>& describe() const noexcept = 0;

    //! init() 之后【实际生效】的配置（引擎解析后的 batch / dstH / dstW 等）。
    //! 与传入 init() 的 cfg 可能不同：静态形状会被纠正为引擎声明值。
    [[nodiscard]] virtual const core::ModelConfig& config() const noexcept = 0;

protected:
    IModel() = default;
};

//! setBatch 的容量护栏 —— 所有任务（det / seg / kpt / cls）共用一处。
//!
//! 为什么必须有：模型的每一块 staging / 输出显存都按【引擎解析后的 batch】分配，
//! 而 setBatch 收到的 Batch 来自调用方。views 多于容量时，preprocess / decode
//! 会按 views.size() 驱动 kernel → 越过已分配的显存写入（UB，且不报错）。
//! 因此这里按"配置与引擎能力不符 → 显式失败"的口径直接抛，既不截断也不静默。
//!
//! 这是**基类收口**：新模型只要用本函数取 batch，就自动获得护栏
//!（对照 InferencePool::submit —— 生产路径上还有一道更靠前的同款检查）。
//!
//! @param model 模型自身（用于取 init() 后的生效配置）
//! @param batch 本批输入
//! @param who   模型名（进日志 / 异常信息）
//! @return 本次实际 batch（== batch.views.size()），便于调用方一行赋值
inline int requireBatchCapacity(const IModel& model, const core::Batch& batch,
                                const char* who)
{
    const int capacity = model.config().batchSize;
    const int n = static_cast<int>(batch.views.size());
    if (capacity > 0 && n > capacity)
    {
        const std::string msg =
            std::string(who) + ": batch of " + std::to_string(n) +
            " images exceeds model capacity " + std::to_string(capacity) +
            " (buffers are sized by the engine-resolved batch; raise [input]"
            " batch_size within the engine profile instead)";
        TRT_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }
    return n;
}

}  // namespace trt_alpha