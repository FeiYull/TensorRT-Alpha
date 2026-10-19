add_executable(sample_yolov8_cls sample_yolov8_cls.cpp)

target_link_libraries(sample_yolov8_cls
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
