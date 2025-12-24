message(STATUS "Using RPC backend")

option(GGML_RPC_RDMA "Enable RDMA transport for RPC backend" OFF)

set(RPC_SOURCES
    ggml-rpc.cpp
    rpc-transport.cpp
)

ggml_add_backend_library(ggml-rpc ${RPC_SOURCES})

add_executable(rpc-transport-test test-rpc-transport.cpp)
target_link_libraries(rpc-transport-test PRIVATE ggml-rpc)

add_executable(rpc-transport-config-test test-rpc-config.cpp)
target_link_libraries(rpc-transport-config-test PRIVATE ggml-rpc)

# Compression test
add_executable(rpc-compression-test test-rpc-compression.cpp)
target_link_libraries(rpc-compression-test PRIVATE ggml-rpc)

# Try to find zstd and enable if available
find_package(ZSTD QUIET)
if (ZSTD_FOUND)
    message(STATUS "Found zstd; enabling RPC zstd compression support")
    target_link_libraries(ggml-rpc PRIVATE ZSTD::zstd)
    target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_HAVE_ZSTD=1)
endif()

if (GGML_RPC_RDMA)
    message(STATUS "RPC RDMA transport enabled: will try to link libibverbs and librdmacm")
    if (WIN32)
        message(FATAL_ERROR "RDMA transport is only supported on Linux in this build")
    endif()
    target_link_libraries(ggml-rpc PRIVATE ibverbs rdmacm)
endif()

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()
