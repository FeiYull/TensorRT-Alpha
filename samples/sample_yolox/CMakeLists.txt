add_executable(sample_yolox sample_yolox.cpp)

target_link_libraries(sample_yolox
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)