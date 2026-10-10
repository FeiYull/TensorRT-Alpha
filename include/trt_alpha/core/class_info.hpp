// =============================================================================
//  trt_alpha :: core :: class_info
// -----------------------------------------------------------------------------
//  ClassInfo -- display information for one class (name + colour).
//
//  Used by:
//    * the renderer to draw labels (name + confidence)
//    * the renderer to pick colours (boxes / masks)
//
//  Source:
//    * read from a TXT file (one "name R G B" per line; the line number is the
//      label)
//    * example file: data/classes/coco80.txt
//
//  Colour convention:
//    * the file holds RGB (0-255)
//    * the struct holds RGB
//    * the renderer converts to BGR internally (the OpenCV convention)
// =============================================================================
#pragma once

#include <cstdint>
#include <string>

namespace trt_alpha::core {

struct ClassInfo
{
    std::string name;          //!< class name ("person" / "bicycle" / ...)
    std::uint8_t r = 0;        //!< red (0-255)
    std::uint8_t g = 0;        //!< green (0-255)
    std::uint8_t b = 0;        //!< blue (0-255)
};

}  // namespace trt_alpha::core
