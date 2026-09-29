add_executable(test_yolov8_pose test_yolov8_pose.cpp)

target_link_libraries(test_yolov8_pose
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)