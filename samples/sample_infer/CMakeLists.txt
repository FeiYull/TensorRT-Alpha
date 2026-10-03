add_executable(sample_infer sample_infer.cpp)

target_link_libraries(sample_infer
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)