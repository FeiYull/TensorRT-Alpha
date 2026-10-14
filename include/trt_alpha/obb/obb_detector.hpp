// =============================================================================
//  trt_alpha :: obb :: obb_detector
// -----------------------------------------------------------------------------
//  IObbDetector -- the oriented-bounding-box task base class (inherits IModel
//  and adds an obbs() accessor).
//
//  Same letterbox preprocessing as detection. Postprocessing differs in two
//  ways: the decoded boxes carry an extra angle channel, and the NMS is the
//  probiou variant (rotated IoU) instead of axis-aligned IoU.
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/obb/types.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_alpha::obb {

class IObbDetector : public IModel
{
public:
    [[nodiscard]] const std::vector<std::vector<ObbDetection>>& obbs() const noexcept
    {
        return m_obbs;
    }

    void commitResult(core::BatchResult& out) override
    {
        out.obbs = std::move(m_obbs);
        m_obbs.clear();
    }

    [[nodiscard]] const core::ModelConfig& config() const noexcept override { return m_cfg; }

protected:
    void loadCommonConfig(const core::ModelConfig& cfg)
    {
        m_cfg           = cfg;
        m_numClass      = cfg.getInt  ("num_class",   m_numClass);
        m_confThreshold = cfg.getFloat("conf_thresh", m_confThreshold);
        m_iouThreshold  = cfg.getFloat("iou_thresh",  m_iouThreshold);
        m_topK          = cfg.getInt  ("top_k",       m_topK);
        m_normScale     = cfg.getFloat("scale",       m_normScale);
        m_padValue      = cfg.getFloat("pad_value",   m_padValue);
    }

    core::ModelConfig m_cfg;
    int   m_numClass      = 15;      //!< DOTA v1 (the OBB default)
    float m_confThreshold = 0.25f;
    float m_iouThreshold  = 0.7f;
    int   m_topK          = 300;
    float m_normScale     = 255.f;
    float m_padValue      = 114.f;

    std::vector<std::vector<ObbDetection>> m_obbs;
};

}  // namespace trt_alpha::obb
