# cmake/GL.cmake
# -----------------------------------------------------------------------------
#  可选 OpenGL 渲染后端（GLFW + 系统 GL + CUDA-GL interop）
#
#  和 OpenCV 一样是【可选后端】，默认 OFF —— 新增的第三方依赖不该成为所有人
#  的构建前提。开启方式（不需要任何环境变量）：
#
#      cmake -S . -B build -DTRT_ALPHA_WITH_GL=ON -DGLFW_ROOT=D:/ThirdParty/glfw
#
#  关掉时本文件只产出一个空的 INTERFACE target，让源代码里无条件的
#  target_link_libraries(... gl::glfw) 继续成立。
# -----------------------------------------------------------------------------

option(TRT_ALPHA_WITH_GL
       "Build the OpenGL render backend (GLFW + CUDA-GL interop)"
       OFF)

if(NOT TRT_ALPHA_WITH_GL)
  message(STATUS "OpenGL: OFF (-DTRT_ALPHA_WITH_GL=ON -DGLFW_ROOT=<dir> to enable)")
  if(NOT TARGET gl::glfw)
    add_library(gl::glfw INTERFACE IMPORTED)
  endif()
  return()
endif()

# ---- 定位 GLFW：只认 CMake 变量，不读环境变量 -------------------------------
if(NOT DEFINED GLFW_ROOT OR GLFW_ROOT STREQUAL "")
  if(WIN32)
    set(GLFW_ROOT "D:/ThirdParty/glfw" CACHE PATH "GLFW root (include/ + lib-vc2022/)" FORCE)
  else()
    set(GLFW_ROOT "/usr/local" CACHE PATH "GLFW root" FORCE)
  endif()
endif()

find_path(TRT_ALPHA_GLFW_INCLUDE_DIR GLFW/glfw3.h
          HINTS "${GLFW_ROOT}/include")
find_library(TRT_ALPHA_GLFW_LIBRARY
             NAMES glfw3 glfw
             HINTS "${GLFW_ROOT}/lib-vc2022" "${GLFW_ROOT}/lib")

if(NOT TRT_ALPHA_GLFW_INCLUDE_DIR OR NOT TRT_ALPHA_GLFW_LIBRARY)
  message(FATAL_ERROR
    "TRT_ALPHA_WITH_GL=ON but GLFW was not found under '${GLFW_ROOT}'.\n"
    "  expected: ${GLFW_ROOT}/include/GLFW/glfw3.h\n"
    "            ${GLFW_ROOT}/lib-vc2022/glfw3.lib  (Windows)\n"
    "            ${GLFW_ROOT}/lib/libglfw3.a         (Linux)\n"
    "  pass -DGLFW_ROOT=<your glfw dir>")
endif()

message(STATUS "[GLFW] include : ${TRT_ALPHA_GLFW_INCLUDE_DIR}")
message(STATUS "[GLFW] library : ${TRT_ALPHA_GLFW_LIBRARY}")

if(NOT TARGET gl::glfw)
  add_library(gl::glfw INTERFACE IMPORTED)
  set_target_properties(gl::glfw PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TRT_ALPHA_GLFW_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${TRT_ALPHA_GLFW_LIBRARY}")

  if(WIN32)
    # 官方 prebuilt 的 glfw3.lib 是静态库，窗口/上下文代码依赖这些系统库。
    set_property(TARGET gl::glfw APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES opengl32 gdi32 user32 shell32)
  else()
    find_package(OpenGL REQUIRED)
    set_property(TARGET gl::glfw APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES OpenGL::GL ${CMAKE_DL_LIBS})
  endif()
endif()

# 唯一一处定义该宏（与 TRT_ALPHA_WITH_OPENCV 同一套约定）
target_compile_definitions(gl::glfw INTERFACE TRT_ALPHA_WITH_GL=1)
