add_executable(sample_yolov4 sample_yolov4.cpp)

target_link_libraries(sample_yolov4
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)