add_executable(sample_yolov7 sample_yolov7.cpp)

target_link_libraries(sample_yolov7
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)