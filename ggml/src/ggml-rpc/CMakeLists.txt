message(STATUS "Using RPC backend")

option(GGML_RPC_RDMA "Enable RDMA transport for RPC backend" OFF)

set(RPC_SOURCES
    ggml-rpc.cpp
    rpc-transport.cpp
)

ggml_add_backend_library(ggml-rpc ${RPC_SOURCES})

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

# RDMA auto-detection (Linux only, requires libibverbs)
if (NOT WIN32 AND NOT APPLE)
    find_library(IBVERBS_LIB ibverbs)
    if (IBVERBS_LIB)
        option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" ON)
    else()
        option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" OFF)
    endif()
else()
    set(GGML_RPC_RDMA OFF CACHE BOOL "RDMA not available on this platform" FORCE)
endif()

if (GGML_RPC_RDMA)
    if (NOT IBVERBS_LIB)
        find_library(IBVERBS_LIB ibverbs REQUIRED)
    endif()
    target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_RDMA)
    target_link_libraries(ggml-rpc PRIVATE ${IBVERBS_LIB})
    message(STATUS "  RDMA transport enabled (auto-detected)")
else()
    message(STATUS "  RDMA transport disabled")
endif()
