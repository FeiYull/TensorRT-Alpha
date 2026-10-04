// =============================================================================
//  trt_alpha :: kpt :: keypointer
// -----------------------------------------------------------------------------
//  IKeypointer —— 姿态估计任务基类（继承 IModel，加 keypoints() 访问器）。
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/kpt/types.hpp"

#include <cstdio>
#include <string>
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
    //! 从 cfg 读通用参数（一次实现，所有姿态模型共用）。
    void loadCommonConfig(const core::ModelConfig& cfg)
    {
        m_cfg           = cfg;
        m_numClass      = cfg.getInt  ("num_class",   m_numClass);
        m_confThreshold = cfg.getFloat("conf_thresh", m_confThreshold);
        m_iouThreshold  = cfg.getFloat("iou_thresh",  m_iouThreshold);
        m_topK          = cfg.getInt  ("top_k",       m_topK);
        m_normScale     = cfg.getFloat("scale",       m_normScale);
        m_padValue      = cfg.getFloat("pad_value",   m_padValue);

        // mean / std：逗号分隔的三元组
        const std::string meanStr = cfg.getString("mean", "");
        if (!meanStr.empty()) {
            float v[3];
            if (std::sscanf(meanStr.c_str(), "%f,%f,%f", &v[0], &v[1], &v[2]) == 3) {
                m_normMean[0] = v[0]; m_normMean[1] = v[1]; m_normMean[2] = v[2];
            }
        }
        const std::string stdStr = cfg.getString("std", "");
        if (!stdStr.empty()) {
            float v[3];
            if (std::sscanf(stdStr.c_str(), "%f,%f,%f", &v[0], &v[1], &v[2]) == 3) {
                m_normStd[0] = v[0]; m_normStd[1] = v[1]; m_normStd[2] = v[2];
            }
        }
    }

    //! 所有姿态模型共用的成员。
    //! iou_thresh 默认 0.7（pose 专属，不是 0.45）。
    core::ModelConfig m_cfg;
    int   m_numClass      = 80;
    float m_confThreshold = 0.25f;
    float m_iouThreshold  = 0.7f;
    int   m_topK          = 300;
    float m_normScale     = 255.f;
    float m_normMean[3]   = {0.f, 0.f, 0.f};
    float m_normStd[3]    = {1.f, 1.f, 1.f};
    float m_padValue      = 114.f;

    std::vector<std::vector<KeypointResult>> m_keypoints;
};

}  // namespace trt_alpha::kpt