// =============================================================================
//  trt_alpha :: kernels :: candidates
// -----------------------------------------------------------------------------
//  Deterministic ordering of the decoded candidate list (host side).
//
//  The legacy decode kernels hand out output slots with atomicAdd, so the slot
//  order follows thread arrival order -- which differs from run to run for the
//  exact same input.  Reading the array as-is therefore made the rendered
//  result non-reproducible (same binary, same image, different bytes).
//
//  Everything downstream only cares about the *set* of survivors, but the
//  floating-point accumulation order in the mask composite and the paint order
//  of overlapping boxes / labels both follow the array order.  This helper
//  yields a total order that does not depend on arrival order, and that is also
//  what ultralytics produces (NMS returns candidates in descending score).
// =============================================================================
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace trt_alpha::kernels {

//! 候选行布局（与 postprocess.hpp 的对象缓冲一致）：
//!   row[0]                    有效条数
//!   row[1 + i*stride + 0..3]  left top right bottom（网络输入坐标）
//!   row[1 + i*stride + 4]     confidence
//!   row[1 + i*stride + 5]     label
//!   row[1 + i*stride + 6]     keep（0 表示被 NMS 淘汰）
//!
//! 返回所有 keep != 0 的候选下标，按
//!   置信度降序 → left/x1 升序 → top/y1 → right/x2 → bottom/y2 → label
//! 排列。前 5 个键相同即视为同一个框，所以比较器是严格全序，`std::sort`
//! 的稳定性无关紧要。
inline std::vector<int> orderedCandidates(const float* row, int count, int stride)
{
    std::vector<int> idx;
    if (row == nullptr || count <= 0 || stride < 7)
    {
        return idx;
    }
    idx.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        if (row[1 + static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + 6]
            != 0.0f)
        {
            idx.push_back(i);
        }
    }
    std::sort(idx.begin(), idx.end(), [row, stride](int a, int b)
    {
        const float* pa = row + 1 + static_cast<std::size_t>(a) * static_cast<std::size_t>(stride);
        const float* pb = row + 1 + static_cast<std::size_t>(b) * static_cast<std::size_t>(stride);
        if (pa[4] != pb[4]) { return pa[4] > pb[4]; }
        for (int k = 0; k < 4; ++k)
        {
            if (pa[k] != pb[k]) { return pa[k] < pb[k]; }
        }
        return pa[5] < pb[5];
    });
    return idx;
}

}  // namespace trt_alpha::kernels
