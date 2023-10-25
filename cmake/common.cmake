# set
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
# find thirdparty
find_package(CUDA REQUIRED)
list(APPEND ALL_LIBS 
  ${CUDA_LIBRARIES} 
  ${CUDA_cublas_LIBRARY} 
  ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} ${CUDA_nppidei_LIBRARY} ${CUDA_nppial_LIBRARY})

# include cuda's header
list(APPEND INCLUDE_DRIS ${CUDA_INCLUDE_DIRS})
# message(FATAL_ERROR "CUDA_npp_LIBRARY: ${CUDA_npp_LIBRARY}")

# gather TensorRT lib
#set(TensorRT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../TensorRT)
#set(TensorRT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../TensorRT-8.2.1.8)
# my tensorrt's path!
#set(TensorRT_ROOT /root/TensorRT-8.2.1.8)
#set(TensorRT_ROOT /root/TensorRT-Plugin)
set(TensorRT_ROOT /root/TensorRT-8.4.2.4)

find_library(TRT_NVINFER NAMES nvinfer HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVINFER_PLUGIN NAMES nvinfer_plugin HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVONNX_PARSER NAMES nvonnxparser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVCAFFE_PARSER NAMES nvcaffe_parser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${TensorRT_ROOT} PATH_SUFFIXES include)
list(APPEND ALL_LIBS ${TRT_NVINFER} ${TRT_NVINFER_PLUGIN} ${TRT_NVONNX_PARSER} ${TRT_NVCAFFE_PARSER})

# include tensorrt's headers
list(APPEND INCLUDE_DRIS ${TENSORRT_INCLUDE_DIR})

# include tensorrt's sample/common headers
#set(SAMPLES_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common)
#set(SAMPLES_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(SAMPLES_COMMON_DIR ${TensorRT_ROOT}/samples/common)
list(APPEND INCLUDE_DRIS ${SAMPLES_COMMON_DIR})
message(STATUS ***INCLUDE_DRIS*** = ${INCLUDE_DRIS})
message(STATUS "ALL_LIBS: ${ALL_LIBS}")
