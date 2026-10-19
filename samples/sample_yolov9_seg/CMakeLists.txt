add_executable(sample_yolov9_seg sample_yolov9_seg.cpp)

target_link_libraries(sample_yolov9_seg
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
