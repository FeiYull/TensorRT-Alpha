// =============================================================================
//  trt_alpha :: core :: layout（实现）
// -----------------------------------------------------------------------------
//  Layout 的常用常量定义。
//  放在独立的 TU 里是因为类内是自身的不完整类型，无法就地初始化静态成员。
//  初始化式是常量表达式 → 编译期常量初始化，不存在静态初始化顺序问题。
// =============================================================================
#include "trt_alpha/core/layout.hpp"

namespace trt_alpha::core {

const Layout Layout::NCHW {"NCHW"};
const Layout Layout::NHWC {"NHWC"};
const Layout Layout::NCDHW{"NCDHW"};
const Layout Layout::NDHWC{"NDHWC"};
const Layout Layout::CHW  {"CHW"};
const Layout Layout::HWC  {"HWC"};

}  // namespace trt_alpha::core
