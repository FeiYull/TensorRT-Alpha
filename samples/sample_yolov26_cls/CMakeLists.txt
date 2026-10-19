add_executable(sample_yolov26_cls sample_yolov26_cls.cpp)

target_link_libraries(sample_yolov26_cls
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
