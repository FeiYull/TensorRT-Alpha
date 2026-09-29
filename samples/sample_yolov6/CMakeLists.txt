add_executable(sample_yolov6 sample_yolov6.cpp)

target_link_libraries(sample_yolov6
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)