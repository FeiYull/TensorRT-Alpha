add_executable(sample_yolov5 sample_yolov5.cpp)

target_link_libraries(sample_yolov5
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)