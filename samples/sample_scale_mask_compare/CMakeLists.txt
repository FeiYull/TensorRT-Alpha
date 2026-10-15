add_executable(sample_scale_mask_compare sample_scale_mask_compare.cpp)

target_link_libraries(sample_scale_mask_compare
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
)
