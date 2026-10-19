add_executable(sample_yolov26_sem sample_yolov26_sem.cpp)

target_link_libraries(sample_yolov26_sem
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
