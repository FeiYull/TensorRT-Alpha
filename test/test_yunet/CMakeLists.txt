add_executable(test_yunet test_yunet.cpp)

target_link_libraries(test_yunet
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)