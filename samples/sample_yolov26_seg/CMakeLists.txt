add_executable(sample_yolov26_seg sample_yolov26_seg.cpp)

target_link_libraries(sample_yolov26_seg
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
