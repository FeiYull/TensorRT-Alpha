# cmake/CUDA.cmake
# 仅用于写 .cu 测试核函数等独立场景
# TensorRT 场景不需要 include 本文件

if(NOT DEFINED CUDAToolkit_ROOT OR CUDAToolkit_ROOT STREQUAL "")
  if(WIN32)
    if(DEFINED ENV{CUDA_PATH})
      set(CUDAToolkit_ROOT "$ENV{CUDA_PATH}" CACHE PATH "CUDA root" FORCE)
    endif()
  else()
    foreach(_p "/usr/local/cuda" "/usr/local/cuda-12.9" "/opt/cuda")
      if(EXISTS "${_p}/include/cuda_runtime.h")
        set(CUDAToolkit_ROOT "${_p}" CACHE PATH "CUDA root" FORCE)
        break()
      endif()
    endforeach()
  endif()
endif()

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

message(STATUS "[CUDA] version : ${CUDAToolkit_VERSION}")
message(STATUS "[CUDA] include : ${CUDAToolkit_INCLUDE_DIRS}")