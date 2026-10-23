# cmake/OpenCV.cmake
# 跨平台封装 OpenCV（CONFIG 模式，读取官方 OpenCVConfig.cmake）
#
# 🔴 OpenCV 是【可选后端】：-DTRT_ALPHA_WITH_OPENCV=OFF 时整套 OpenCV 依赖消失，
#    改用 GLFW + OpenGL（见 cmake/GL.cmake）。默认 ON，老用户构建零变化。

option(TRT_ALPHA_WITH_OPENCV
       "Build the OpenCV backend (data source + renderer). OFF removes the dependency entirely"
       ON)

if(NOT TRT_ALPHA_WITH_OPENCV)
  message(STATUS "OpenCV: OFF (-DTRT_ALPHA_WITH_OPENCV=ON to enable)")
  # 空的 INTERFACE target：让无条件写着的 target_link_libraries(... ocv::ocv)
  # 继续可用，而 OFF 时不携带任何 include / lib / 宏。
  if(NOT TARGET ocv::ocv)
    add_library(ocv::ocv INTERFACE IMPORTED)
  endif()
  return()
endif()

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

# 唯一一处定义该宏：所有链接 trt_alpha_core 的 target 都继承到，
# 于是 source_factory / renderer_factory 里的 #if 判断处处一致。
target_compile_definitions(ocv::ocv INTERFACE TRT_ALPHA_WITH_OPENCV=1)
