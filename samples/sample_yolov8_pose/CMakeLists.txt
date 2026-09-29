add_executable(sample_yolov8_pose sample_yolov8_pose.cpp)

target_link_libraries(sample_yolov8_pose
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)