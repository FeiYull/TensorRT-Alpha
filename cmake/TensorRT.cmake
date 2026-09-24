# cmake/TensorRT.cmake
# 跨平台封装 TensorRT 10.x（兼容 8.x）
# TensorRT 依赖 CUDA Runtime，所以本文件里会调用 find_package(CUDAToolkit)
# 默认路径：Windows = D:/ThirdParty/TensorRT-10.16.1.11
#           Linux   = /usr/local/TensorRT
# 可通过 -DTensorRT_ROOT=... 覆盖

# ============================================================
# 1. 定位 TensorRT 根目录
# ============================================================
if(NOT DEFINED TensorRT_ROOT OR TensorRT_ROOT STREQUAL "")
  if(WIN32)
    set(_trt_defaults
        "D:/ThirdParty/TensorRT-10.16.1.11"
        "$ENV{TensorRT_ROOT}")
  else()
    set(_trt_defaults
        "/usr/local/TensorRT"
        "/opt/TensorRT"
        "$ENV{TensorRT_ROOT}")
  endif()

  foreach(_p ${_trt_defaults})
    if(_p AND EXISTS "${_p}/include/NvInfer.h")
      set(TensorRT_ROOT "${_p}" CACHE PATH "TensorRT root dir" FORCE)
      break()
    endif()
  endforeach()
endif()

# ============================================================
# 2. 定位 TensorRT 头文件
# ============================================================
find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${TensorRT_ROOT}
  PATH_SUFFIXES include)

# ============================================================
# 3. 定位 TensorRT 核心库
#    TRT 10 库名带 _10 后缀，8.x 不带，所以两种都写
# ============================================================
find_library(TensorRT_NVINFER_LIB
  NAMES nvinfer_10 nvinfer
  HINTS ${TensorRT_ROOT}
  PATH_SUFFIXES lib lib/x64)

find_library(TensorRT_NVINFER_PLUGIN_LIB
  NAMES nvinfer_plugin_10 nvinfer_plugin
  HINTS ${TensorRT_ROOT}
  PATH_SUFFIXES lib lib/x64)

find_library(TensorRT_NVONNXPARSER_LIB
  NAMES nvonnxparser_10 nvonnxparser
  HINTS ${TensorRT_ROOT}
  PATH_SUFFIXES lib lib/x64)

# ============================================================
# 4. 可选：cuDNN
#    TensorRT 10 不再强制依赖，但老模型/老 plan 可能用到，
#    所以做成“找到就加，找不到不报错”
# ============================================================
find_library(TensorRT_CUDNN_LIB
  NAMES cudnn
  HINTS ${TensorRT_ROOT}
  PATH_SUFFIXES lib lib/x64
  PATHS "$ENV{CUDA_PATH}/lib/x64"
  NO_DEFAULT_PATH)

if(NOT TensorRT_CUDNN_LIB)
  # 回退到系统默认路径（Linux 常见）
  find_library(TensorRT_CUDNN_LIB NAMES cudnn)
endif()

# ============================================================
# 5. 关键：TensorRT 依赖 CUDA，必须找到 CUDA Toolkit
#    - 提供 cuda_runtime_api.h 等头文件（否则编译报 C1083）
#    - 提供 CUDA::cudart 运行时库
#    find_package(CUDAToolkit) 是 CMake 3.17+ 自带模块，跨平台
#    Windows 靠 CUDA_PATH 环境变量
#    Linux  靠 /usr/local/cuda
# ============================================================
find_package(CUDAToolkit REQUIRED)

# ============================================================
# 6. 检查 TensorRT 必需项
# ============================================================
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIB
    TensorRT_NVONNXPARSER_LIB)

# ============================================================
# 7. 封装成 trt::trt
#    - INTERFACE_INCLUDE_DIRECTORIES 同时包含 TRT 和 CUDA 的头文件路径
#    - INTERFACE_LINK_LIBRARIES 包含 TRT 的 .lib/.so + CUDA::cudart
# ============================================================
if(TensorRT_FOUND AND NOT TARGET trt::trt)
  add_library(trt::trt INTERFACE IMPORTED)

  # ---- 链接库列表 ----
  set(_trt_libs
      "${TensorRT_NVINFER_LIB}"
      "${TensorRT_NVONNXPARSER_LIB}")

  # 插件库（可选，代码里用 initLibNvInferPlugins 等才需要）
  if(TensorRT_NVINFER_PLUGIN_LIB)
    list(APPEND _trt_libs "${TensorRT_NVINFER_PLUGIN_LIB}")
  endif()

  # cuDNN（可选）
  if(TensorRT_CUDNN_LIB)
    list(APPEND _trt_libs "${TensorRT_CUDNN_LIB}")
  endif()

  # CUDA 运行时（必需）
  list(APPEND _trt_libs CUDA::cudart)

  # ---- 头文件路径 ----
  # 把 TensorRT 和 CUDA 的头文件目录合并
  set(_trt_includes
      "${TensorRT_INCLUDE_DIR}"
      "${CUDAToolkit_INCLUDE_DIRS}")

  set_target_properties(trt::trt PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_trt_includes}"
    INTERFACE_LINK_LIBRARIES      "${_trt_libs}")
endif()

# ============================================================
# 8. 打印结果，方便排查
# ============================================================
if(TensorRT_FOUND)
  message(STATUS "[TensorRT] root       : ${TensorRT_ROOT}")
  message(STATUS "[TensorRT] include    : ${TensorRT_INCLUDE_DIR}")
  message(STATUS "[TensorRT] nvinfer    : ${TensorRT_NVINFER_LIB}")
  message(STATUS "[TensorRT] onnxparser : ${TensorRT_NVONNXPARSER_LIB}")
  if(TensorRT_NVINFER_PLUGIN_LIB)
    message(STATUS "[TensorRT] plugin     : ${TensorRT_NVINFER_PLUGIN_LIB}")
  endif()
  if(TensorRT_CUDNN_LIB)
    message(STATUS "[TensorRT] cudnn      : ${TensorRT_CUDNN_LIB}")
  else()
    message(STATUS "[TensorRT] cudnn      : not found (optional)")
  endif()
  message(STATUS "[TensorRT] CUDA       : ${CUDAToolkit_VERSION} @ ${CUDAToolkit_LIBRARY_DIR}")
endif()