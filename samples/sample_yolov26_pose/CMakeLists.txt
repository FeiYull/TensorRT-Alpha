add_executable(sample_yolov26_pose sample_yolov26_pose.cpp)

target_link_libraries(sample_yolov26_pose
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
