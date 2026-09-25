add_executable(test_bounded_queue test_bounded_queue.cpp)

# 纯头文件，不需要 trt::trt / ocv::ocv
target_link_libraries(test_bounded_queue PRIVATE trt_alpha_core)