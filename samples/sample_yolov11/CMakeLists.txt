add_executable(sample_yolov11 sample_yolov11.cpp)

target_link_libraries(sample_yolov11
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
