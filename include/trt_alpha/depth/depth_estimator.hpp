// =============================================================================
//  trt_alpha :: depth :: depth_estimator
// -----------------------------------------------------------------------------
//  IDepthEstimator -- the monocular-depth task base class (inherits IModel and
//  adds a depths() accessor).
//
//  Preprocessing is the same letterbox as detection. Postprocessing is NOT
//  NMS: the engine already emits a 640x640 metre-valued map, so the only step
//  left is the inverse letterbox (crop the padding away, then bilinear-resample
//  back onto the source image).
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/depth/types.hpp"

#include <utility>
#include <vector>

namespace trt_alpha::depth {

class IDepthEstimator : public IModel
{
public:
    [[nodiscard]] const std::vector<DepthMap>& depths() const noexcept { return m_depths; }

    void commitResult(core::BatchResult& out) override
    {
        out.depths = std::move(m_depths);
        m_depths.clear();
    }

    [[nodiscard]] const core::ModelConfig& config() const noexcept override { return m_cfg; }

protected:
    void loadCommonConfig(const core::ModelConfig& cfg)
    {
        m_cfg       = cfg;
        m_normScale = cfg.getFloat("scale",     m_normScale);
        m_padValue  = cfg.getFloat("pad_value", m_padValue);
    }

    core::ModelConfig m_cfg;
    float m_normScale = 255.f;
    float m_padValue  = 114.f;

    std::vector<DepthMap> m_depths;
};

}  // namespace trt_alpha::depth
