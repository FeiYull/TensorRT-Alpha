// =============================================================================
//  trt_alpha :: sem :: semantic_segmentor
// -----------------------------------------------------------------------------
//  ISemanticSegmentor -- the semantic-segmentation task base class (inherits
//  IModel and adds a semantics() accessor).
//
//  The YOLO26 sem engine emits an already-argmax'ed uint8 class map, so the
//  only postprocessing is the inverse letterbox: crop the padding away and
//  resample back onto the source image with NEAREST (resampling class ids with
//  bilinear would invent classes that never existed).
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/sem/types.hpp"

#include <utility>
#include <vector>

namespace trt_alpha::sem {

class ISemanticSegmentor : public IModel
{
public:
    [[nodiscard]] const std::vector<SemanticMask>& semantics() const noexcept { return m_semantics; }

    void commitResult(core::BatchResult& out) override
    {
        out.semantics = std::move(m_semantics);
        m_semantics.clear();
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

    std::vector<SemanticMask> m_semantics;
};

}  // namespace trt_alpha::sem
