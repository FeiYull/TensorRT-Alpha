add_executable(sample_yolov10 sample_yolov10.cpp)

target_link_libraries(sample_yolov10
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
