add_executable(test_model_registry test_model_registry.cpp)

# model.hpp include <NvInfer.h>（通过 engine.hpp），需要 trt::trt 提供 include 路径
target_link_libraries(test_model_registry PRIVATE trt_alpha_core trt::trt)