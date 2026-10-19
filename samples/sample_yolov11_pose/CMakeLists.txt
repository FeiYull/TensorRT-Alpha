add_executable(sample_yolov11_pose sample_yolov11_pose.cpp)

target_link_libraries(sample_yolov11_pose
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
