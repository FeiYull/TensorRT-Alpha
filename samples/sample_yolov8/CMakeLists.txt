add_executable(sample_yolov8 sample_yolov8.cpp)

target_link_libraries(sample_yolov8
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)