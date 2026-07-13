set(GGML_SOURCES_HTP
    dsprpc_interface.cc
    dsprpc_interface.h
    ggml-htp-impl.h
    ggml-htp.cc
    htp-cpu-impl.c
    htp-ops.cc
    htp-ops.h
    message.h
    op_reg.h
    rpcmem_mapper.cc
    rpcmem_mapper.h
)

ggml_add_backend_library(ggml-htp ${GGML_SOURCES_HTP})

if (GGML_OPENCL AND NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml-htp PRIVATE GGML_HTP_OPENCL_SIDECAR)
    target_link_libraries(ggml-htp PRIVATE ggml-opencl)
endif()

if (UNIX AND NOT APPLE)
    target_link_libraries(ggml-htp PRIVATE dl)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Android")
    target_link_libraries(ggml-htp PRIVATE ggml-cpu)
endif()
