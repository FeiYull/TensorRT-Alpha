// =============================================================================
//  trt_alpha :: cls :: classifier
// -----------------------------------------------------------------------------
//  IClassifier -- the classification task base class (inherits IModel and adds
//  a classifications() accessor).
//
//  Unlike detection, classification has no letterbox and no NMS: the only
//  postprocessing is a top-k over the (already softmax'ed) score vector, and
//  the preprocessing is a PIL-faithful resize + centre crop. Neither m_iou nor
//  m_conf exists here on purpose.
// =============================================================================
#pragma once

#include "trt_alpha/cls/types.hpp"
#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_alpha::cls {

class IClassifier : public IModel
{
public:
    [[nodiscard]] const std::vector<std::vector<ClassScore>>& classifications() const noexcept
    {
        return m_classifications;
    }

    void commitResult(core::BatchResult& out) override
    {
        out.classifications = std::move(m_classifications);
        m_classifications.clear();
    }

    [[nodiscard]] const core::ModelConfig& config() const noexcept override { return m_cfg; }

protected:
    void loadCommonConfig(const core::ModelConfig& cfg)
    {
        m_cfg       = cfg;
        m_numClass  = cfg.getInt  ("num_class", m_numClass);
        m_topK      = cfg.getInt  ("top_k",     m_topK);
        m_normScale = cfg.getFloat("scale",     m_normScale);
        m_padValue  = cfg.getFloat("pad_value", m_padValue);
    }

    core::ModelConfig m_cfg;
    int   m_numClass  = 1000;      //!< ImageNet-1k by default
    int   m_topK      = 5;
    float m_normScale = 255.f;
    float m_padValue  = 0.f;       //!< classification resize pads with 0 (PIL), not 114

    std::vector<std::vector<ClassScore>> m_classifications;
};

}  // namespace trt_alpha::cls
