// =============================================================================
//  trt_alpha :: seg :: segmentor
// -----------------------------------------------------------------------------
//  ISegmentor —— 分割任务基类（继承 IModel，加 segmentations() 访问器）。
//
//  设计：
//    * 任务基类只约定【输出结构体】，不约定输出内存布局
//    * 具体模型（YoloV8Seg / PPHumanSeg / U2Net）继承本类，自己实现
//      init / setBatch / preprocess / infer / postprocess
//    * commitResult() 由本基类提供默认实现（把 m_segmentations move 到 out）
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/seg/types.hpp"

#include <utility>
#include <vector>

namespace trt_alpha::seg {

class ISegmentor : public IModel
{
public:
    //! 每张图一个 Segmentation 列表；生命周期到下一次 commitResult / reset 为止。
    //! 注意：commitResult() 会 move 走这个列表，之后调用本方法返回空。
    [[nodiscard]] const std::vector<std::vector<Segmentation>>& segmentations() const noexcept
    {
        return m_segmentations;
    }

    //! 默认实现：把 m_segmentations move 到 out.segmentations，本对象清空。
    //! 派生类通常不需要覆盖。
    void commitResult(core::BatchResult& out) override
    {
        out.segmentations = std::move(m_segmentations);
        m_segmentations.clear();
    }

protected:
    std::vector<std::vector<Segmentation>> m_segmentations;
};

}  // namespace trt_alpha::seg