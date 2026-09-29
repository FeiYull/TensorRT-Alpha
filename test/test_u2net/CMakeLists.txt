add_executable(test_u2net test_u2net.cpp)

target_link_libraries(test_u2net
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)