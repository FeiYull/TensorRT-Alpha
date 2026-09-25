add_executable(test_pipeline test_pipeline.cpp)

target_link_libraries(test_pipeline
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)