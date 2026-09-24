# cmake/OpenCV.cmake
# 跨平台封装 OpenCV（CONFIG 模式，读取官方 OpenCVConfig.cmake）

if(NOT DEFINED OpenCV_DIR OR OpenCV_DIR STREQUAL "")
  if(WIN32)
    set(_ocv_defaults
        "D:/ThirdParty/opencv/build/x64/vc16/lib"
        "$ENV{OpenCV_DIR}")
  else()
    set(_ocv_defaults
        "/usr/local/lib/cmake/opencv4"
        "/usr/lib/x86_64-linux-gnu/cmake/opencv4"
        "/usr/lib/cmake/opencv4"
        "$ENV{OpenCV_DIR}")
  endif()

  foreach(_p ${_ocv_defaults})
    if(_p AND EXISTS "${_p}/OpenCVConfig.cmake")
      set(OpenCV_DIR "${_p}" CACHE PATH "OpenCV config dir" FORCE)
      break()
    endif()
  endforeach()
endif()

set(OpenCV_STATIC OFF)
set(OpenCV_CUDA   OFF)

# 最低版本取 4.5，兼容 Ubuntu 22.04 apt 装的 4.5.4 和 Win 上的 4.7.0
find_package(OpenCV 4.5 REQUIRED CONFIG)

message(STATUS "[OpenCV] version : ${OpenCV_VERSION}")
message(STATUS "[OpenCV] dir     : ${OpenCV_DIR}")

if(NOT TARGET ocv::ocv)
  add_library(ocv::ocv INTERFACE IMPORTED)
  set_target_properties(ocv::ocv PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES      "${OpenCV_LIBS}")
endif()