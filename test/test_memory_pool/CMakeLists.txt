add_executable(test_memory_pool test_memory_pool.cpp)

# # 需要 CUDA runtime + trt_alpha_core
# target_link_libraries(test_memory_pool PRIVATE trt_alpha_core)

# logger.hpp include <NvInfer.h>，需要 trt::trt 提供 include 路径
target_link_libraries(test_memory_pool PRIVATE trt_alpha_core trt::trt)