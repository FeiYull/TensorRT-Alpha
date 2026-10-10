// =============================================================================
//  trt_alpha :: cls :: types
// -----------------------------------------------------------------------------
//  ClassScore -- one class score (an entry of a classification top-k).
// =============================================================================
#pragma once

namespace trt_alpha::cls {

//! One class score.
struct ClassScore
{
    int label = -1;       //!< class ID
    float score = 0.f;    //!< score (usually a softmax probability)
};

}  // namespace trt_alpha::cls
