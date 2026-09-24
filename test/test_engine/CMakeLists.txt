add_executable(test_engine test_engine.cpp)

# engine.hpp include <NvInfer.h>，需要 trt::trt 提供 include 路径
target_link_libraries(test_engine PRIVATE trt_alpha_core trt::trt)