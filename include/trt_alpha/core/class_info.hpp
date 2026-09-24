// =============================================================================
//  trt_alpha :: core :: class_info
// -----------------------------------------------------------------------------
//  ClassInfo —— 一个类别的展示信息（名字 + 颜色）。
//
//  用途：
//    * 渲染器画标签（名字 + 置信度）
//    * 渲染器取颜色（画框 / 掩码）
//
//  来源：
//    * 从 TXT 文件读（每行 "名字 R G B"，行号 = label）
//    * 文件示例：data/classes/coco80.txt
//
//  颜色约定：
//    * 文件里写 RGB（0-255）
//    * 结构体里存 RGB
//    * 渲染器内部转 BGR（OpenCV 习惯）
// =============================================================================
#pragma once

#include <cstdint>
#include <string>

namespace trt_alpha::core {

struct ClassInfo
{
    std::string name;          //!< 类别名（"person" / "bicycle" / ...）
    std::uint8_t r = 0;        //!< 红（0-255）
    std::uint8_t g = 0;        //!< 绿（0-255）
    std::uint8_t b = 0;        //!< 蓝（0-255）
};

}  // namespace trt_alpha::core