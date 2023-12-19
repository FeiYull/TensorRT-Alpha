cmake_minimum_required(VERSION 3.10)

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

PROJECT(yolov8_seg VERSION 1.0.0 LANGUAGES C CXX CUDA)
message(*********************************************************)
message(STATUS CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR})
message(*********************************************************)

# feiyull
set( CMAKE_CXX_FLAGS "-O3" )
include_directories( "/usr/include/eigen3" )
# in osx and brew install
# include_directories( /usr/local/Cellar/eigen/3.3.3/include/eigen3 )

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)

message(*********************************************************)
message(STATUS TensorRT_ROOT = ${TensorRT_ROOT})
message(*********************************************************)

message(STATUS ${ALL_LIBS})
file(GLOB CPPS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../utils/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../utils/*.cpp
  ${TensorRT_ROOT}/samples/common/logger.cpp 
  ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
  )
list(REMOVE_ITEM CPPS app_yolov8_seg.cpp)
message(*********************************************************)
message(STATUS CPPS = ${CPPS})
message(*********************************************************)

list (LENGTH CPPS length)
message(*********************************************************)
message(STATUS ***length*** = ${length}) 
message(*********************************************************)

find_package(OpenCV REQUIRED)

include_directories(${INCLUDE_DRIS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

message(*********************************************************)
message(STATUS ***INCLUDE_DRIS*** = ${INCLUDE_DRIS})
message(*********************************************************)

message(*********************************************************)
message(STATUS ***OpenCV_INCLUDE_DIRS*** = ${OpenCV_INCLUDE_DIRS})
message(*********************************************************)

add_library(${PROJECT_NAME} SHARED ${CPPS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS} ${OpenCV_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 50 61 72 75)
target_compile_options(${PROJECT_NAME} PUBLIC 
   $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -lineinfo; --use_fast_math --disable-warnings>)
  
add_executable(app_yolov8_seg app_yolov8_seg.cpp)

# NVCC
# target_link_libraries(detect ${PROJECT_NAME} ${ALL_LIBS} ${OpenCV_LIBRARIES} libgflags_nothreads.a)
target_link_libraries(app_yolov8_seg ${PROJECT_NAME} ${ALL_LIBS} ${OpenCV_LIBRARIES} )
