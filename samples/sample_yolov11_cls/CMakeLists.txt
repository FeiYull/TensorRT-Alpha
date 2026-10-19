add_executable(sample_yolov11_cls sample_yolov11_cls.cpp)

target_link_libraries(sample_yolov11_cls
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
