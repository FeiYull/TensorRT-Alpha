add_executable(sample_yolov8_seg sample_yolov8_seg.cpp)

target_link_libraries(sample_yolov8_seg
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)