add_executable(sample_preprocess_compare sample_preprocess_compare.cpp)

target_link_libraries(sample_preprocess_compare
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
)
