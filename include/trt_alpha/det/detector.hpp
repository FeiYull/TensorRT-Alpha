// =============================================================================
//  trt_alpha :: det :: detector
// -----------------------------------------------------------------------------
//  IDetector —— 检测任务基类（继承 IModel，加 detections() 访问器）。
//
//  设计：
//    * 任务基类只约定【输出结构体】，不约定输出内存布局
//    * 具体模型（YoloV8 / EfficientDet / ...）继承本类，自己实现
//      init / setBatch / preprocess / infer / postprocess / commitResult
//    * commitResult() 由本基类提供默认实现（把 m_detections move 到 out）
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/det/types.hpp"

#include <utility>
#include <vector>

namespace trt_alpha::det {

class IDetector : public IModel
{
public:
    //! 每张图一个 Detection 列表；生命周期到下一次 commitResult / reset 为止。
    //! 注意：commitResult() 会 move 走这个列表，之后调用本方法返回空。
    [[nodiscard]] const std::vector<std::vector<Detection>>& detections() const noexcept
    {
        return m_detections;
    }

    //! 默认实现：把 m_detections move 到 out.detections，本对象清空。
    //! 派生类通常不需要覆盖。
    void commitResult(core::BatchResult& out) override
    {
        out.detections = std::move(m_detections);
        m_detections.clear();
    }

protected:
    std::vector<std::vector<Detection>> m_detections;
};

}  // namespace trt_alpha::det