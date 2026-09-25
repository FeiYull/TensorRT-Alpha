add_executable(test_datasource test_datasource.cpp)

target_link_libraries(test_datasource
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
)