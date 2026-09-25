add_executable(test_renderer test_renderer.cpp)

target_link_libraries(test_renderer
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
)