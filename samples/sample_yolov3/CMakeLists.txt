add_executable(sample_yolov3 sample_yolov3.cpp)

target_link_libraries(sample_yolov3
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)