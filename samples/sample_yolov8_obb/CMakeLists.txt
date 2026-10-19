add_executable(sample_yolov8_obb sample_yolov8_obb.cpp)

target_link_libraries(sample_yolov8_obb
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
