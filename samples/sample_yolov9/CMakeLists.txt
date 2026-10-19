add_executable(sample_yolov9 sample_yolov9.cpp)

target_link_libraries(sample_yolov9
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
