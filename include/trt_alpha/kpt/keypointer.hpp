// =============================================================================
//  trt_alpha :: kpt :: keypointer
// -----------------------------------------------------------------------------
//  IKeypointer —— 姿态估计任务基类（继承 IModel，加 keypoints() 访问器）。
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/kpt/types.hpp"

#include <utility>
#include <vector>

namespace trt_alpha::kpt {

class IKeypointer : public IModel
{
public:
    [[nodiscard]] const std::vector<std::vector<KeypointResult>>& keypoints() const noexcept
    {
        return m_keypoints;
    }

    void commitResult(core::BatchResult& out) override
    {
        out.keypoints = std::move(m_keypoints);
        m_keypoints.clear();
    }

protected:
    std::vector<std::vector<KeypointResult>> m_keypoints;
};

}  // namespace trt_alpha::kpt