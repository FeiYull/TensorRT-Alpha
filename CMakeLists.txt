# =============================================================================
#  trt_alpha —— 顶层 CMakeLists
# -----------------------------------------------------------------------------
#  职责：
#    * 声明 C++17 / CUDA 17 标准
#    * 自动探测 CUDA 架构（nvidia-smi）
#    * 收集 src/ 下所有 .cpp / .cu，编成 trt_alpha_core 静态库
#    * 加载依赖（OpenCV / TensorRT，见 cmake/）
#    * 挂上 test/ 子目录
# =============================================================================
cmake_minimum_required(VERSION 3.25)

project(TensorRT_Alpha LANGUAGES CXX CUDA)


# =============================================================================
# CUDA 架构自动选择
# -----------------------------------------------------------------------------
# 优先级：
#   1. -DTRT_ALPHA_CUDA_ARCH=<arch> 用户显式指定
#   2. 本机 nvidia-smi 探测
#   3. 回退主流消费卡列表
#
# 注意：不用 CMAKE_CUDA_ARCHITECTURES 判断"用户是否指定"，因为
# project(... CUDA) 会自动给它设默认值（CUDA 12.9 下是 52）。
# 用自己的 TRT_ALPHA_CUDA_ARCH 作为唯一判据。
# set(... CACHE ... FORCE) 确保覆盖 CMake 隐式设的默认值。
# =============================================================================
if(DEFINED TRT_ALPHA_CUDA_ARCH AND NOT TRT_ALPHA_CUDA_ARCH STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${TRT_ALPHA_CUDA_ARCH}"
        CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "[CUDA] arch from -DTRT_ALPHA_CUDA_ARCH: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    find_program(TRT_ALPHA_NVIDIA_SMI nvidia-smi)
    if(TRT_ALPHA_NVIDIA_SMI)
        execute_process(
            COMMAND "${TRT_ALPHA_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE TRT_ALPHA_GPU_CC
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(TRT_ALPHA_GPU_CC)
        string(REPLACE "." "" TRT_ALPHA_CUDA_ARCH "${TRT_ALPHA_GPU_CC}")
        set(CMAKE_CUDA_ARCHITECTURES "${TRT_ALPHA_CUDA_ARCH}"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "[CUDA] arch detected from local GPU: ${TRT_ALPHA_CUDA_ARCH}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89;120"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "[CUDA] no local GPU detected, fallback: 75;86;89;120")
    endif()
endif()


# =============================================================================
# 语言标准
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# =============================================================================
# 构建类型默认（单配置生成器）
# =============================================================================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()


# =============================================================================
# MSVC 中文乱码：源文件按 UTF-8 编译
# =============================================================================
if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()


# =============================================================================
# 依赖模块目录 + 加载依赖
# =============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(OpenCV)
include(TensorRT)


# =============================================================================
# 统一输出目录
# =============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


# =============================================================================
# trt_alpha_core —— 底座静态库
# -----------------------------------------------------------------------------
# 收集 src/ 下所有 .cpp 和 .cu，编成一个静态库。
# CUDA_SEPARABLE_COMPILATION + CUDA_RESOLVE_DEVICE_SYMBOLS 是 .cu 与 .cpp
# 跨编译单元互相引用时必须的（否则 __cudaRegisterLinkedBinary_* 链接失败）。
# =============================================================================
file(GLOB_RECURSE TRT_ALPHA_SOURCES
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_SOURCE_DIR}/src/*.cu")

add_library(trt_alpha_core STATIC ${TRT_ALPHA_SOURCES})

# 编译期写入源码根路径：out-of-tree 构建时给 Paths 一个兜底
target_compile_definitions(trt_alpha_core PRIVATE
    TRT_ALPHA_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

# Release 构建：TRT_LOG_DEBUG 变成 do {} while(0)，零开销
# Debug 构建：全开（调试方便）
target_compile_definitions(trt_alpha_core PRIVATE
    $<$<CONFIG:Release>:TRT_ALPHA_LOG_MIN_LEVEL=1>
)


set_target_properties(trt_alpha_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(trt_alpha_core
  PUBLIC "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(trt_alpha_core
  PUBLIC  ocv::ocv
  PRIVATE trt::trt)


# =============================================================================
# 测试
# =============================================================================
add_subdirectory(test)