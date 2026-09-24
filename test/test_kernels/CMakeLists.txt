add_executable(test_kernels test_kernels.cpp)

# kernels 是 trt_alpha_core 的一部分；需要 CUDA runtime + trt::trt（logger.hpp 用）
target_link_libraries(test_kernels PRIVATE trt_alpha_core trt::trt)