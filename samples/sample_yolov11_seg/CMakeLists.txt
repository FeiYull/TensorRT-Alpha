add_executable(sample_yolov11_seg sample_yolov11_seg.cpp)

target_link_libraries(sample_yolov11_seg
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
