add_executable(sample_yolov26_depth sample_yolov26_depth.cpp)

target_link_libraries(sample_yolov26_depth
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
