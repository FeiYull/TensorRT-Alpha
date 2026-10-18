add_executable(test_yolov8_pose_old test_yolov8_pose_old.cpp)

target_link_libraries(test_yolov8_pose_old
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
