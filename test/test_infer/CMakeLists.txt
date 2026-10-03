add_executable(test_infer test_infer.cpp)

target_link_libraries(test_infer
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)