// =============================================================================
//  trt_alpha :: det :: detector
// -----------------------------------------------------------------------------
//  IDetector —— 检测任务基类（继承 IModel，加 detections() 访问器）。
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/det/types.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_alpha::det {

class IDetector : public IModel
{
public:
    [[nodiscard]] const std::vector<std::vector<Detection>>& detections() const noexcept
    {
        return m_detections;
    }

    void commitResult(core::BatchResult& out) override
    {
        out.detections = std::move(m_detections);
        m_detections.clear();
    }

    [[nodiscard]] const core::ModelConfig& config() const noexcept override { return m_cfg; }

protected:
    //! 从 cfg 读通用参数（一次实现，所有检测模型共用）。
    void loadCommonConfig(const core::ModelConfig& cfg)
    {
        m_cfg           = cfg;
        m_numClass      = cfg.getInt  ("num_class",   m_numClass);
        m_confThreshold = cfg.getFloat("conf_thresh", m_confThreshold);
        m_iouThreshold  = cfg.getFloat("iou_thresh",  m_iouThreshold);
        m_topK          = cfg.getInt  ("top_k",       m_topK);
        m_normScale     = cfg.getFloat("scale",       m_normScale);
        m_padValue      = cfg.getFloat("pad_value",   m_padValue);

        const std::string meanStr = cfg.getString("mean", "");
        if (!meanStr.empty()) {
            float v[3];
            if (std::sscanf(meanStr.c_str(), "%f,%f,%f", &v[0], &v[1], &v[2]) == 3) {
                m_normMean[0] = v[0]; m_normMean[1] = v[1]; m_normMean[2] = v[2];
            } else {
                throw std::runtime_error(
                    "cfg 'mean' expects 3 comma-separated floats, got '" + meanStr + "'");
            }
        }
        const std::string stdStr = cfg.getString("std", "");
        if (!stdStr.empty()) {
            float v[3];
            if (std::sscanf(stdStr.c_str(), "%f,%f,%f", &v[0], &v[1], &v[2]) == 3) {
                m_normStd[0] = v[0]; m_normStd[1] = v[1]; m_normStd[2] = v[2];
            } else {
                throw std::runtime_error(
                    "cfg 'std' expects 3 comma-separated floats, got '" + stdStr + "'");
            }
        }
    }

    //! 所有检测模型共用的 9 个成员。
    core::ModelConfig m_cfg;
    int   m_numClass      = 80;
    float m_confThreshold = 0.25f;
    float m_iouThreshold  = 0.45f;
    int   m_topK          = 300;
    float m_normScale     = 255.f;
    float m_normMean[3]   = {0.f, 0.f, 0.f};
    float m_normStd[3]    = {1.f, 1.f, 1.f};
    float m_padValue      = 114.f;

    std::vector<std::vector<Detection>> m_detections;
};

}  // namespace trt_alpha::det