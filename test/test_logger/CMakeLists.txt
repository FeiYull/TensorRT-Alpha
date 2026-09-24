add_executable(test_logger test_logger.cpp)

# logger.hpp include <NvInfer.h>，所以要链 trt::trt（拿 include 路径）
target_link_libraries(test_logger PRIVATE trt_alpha_core trt::trt)