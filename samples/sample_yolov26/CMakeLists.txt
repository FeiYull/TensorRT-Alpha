add_executable(sample_yolov26 sample_yolov26.cpp)

target_link_libraries(sample_yolov26
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
