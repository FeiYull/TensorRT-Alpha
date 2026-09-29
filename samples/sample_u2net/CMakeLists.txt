add_executable(sample_u2net sample_u2net.cpp)

target_link_libraries(sample_u2net
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)