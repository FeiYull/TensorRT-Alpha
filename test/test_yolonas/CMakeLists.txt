add_executable(test_yolonas test_yolonas.cpp)

target_link_libraries(test_yolonas
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)