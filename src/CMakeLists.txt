llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-parquet.cpp
    llama-parquet.h
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump
target_precompile_headers (llama PRIVATE models/models.h)

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_PARQUET)
    if (LLAMA_PARQUET_INCLUDE_DIR AND LLAMA_PARQUET_LIBRARIES)
        target_include_directories(llama PRIVATE ${LLAMA_PARQUET_INCLUDE_DIR})
        target_link_libraries(llama PRIVATE ${LLAMA_PARQUET_LIBRARIES})
    else()
        find_package(Arrow CONFIG REQUIRED)
        find_package(Parquet CONFIG REQUIRED)

        if (TARGET Arrow::arrow_shared)
            set(LLAMA_ARROW_TARGET Arrow::arrow_shared)
        elseif (TARGET Arrow::arrow_static)
            set(LLAMA_ARROW_TARGET Arrow::arrow_static)
        else()
            message(FATAL_ERROR "Arrow package has no supported CMake target")
        endif()

        if (TARGET Parquet::parquet_shared)
            set(LLAMA_PARQUET_TARGET Parquet::parquet_shared)
        elseif (TARGET Parquet::parquet_static)
            set(LLAMA_PARQUET_TARGET Parquet::parquet_static)
        else()
            message(FATAL_ERROR "Parquet package has no supported CMake target")
        endif()

        target_link_libraries(llama PRIVATE ${LLAMA_ARROW_TARGET} ${LLAMA_PARQUET_TARGET})
    endif()
    target_compile_definitions(llama PRIVATE LLAMA_USE_PARQUET)
    if (MSVC)
        set_property(SOURCE llama-parquet.cpp APPEND PROPERTY COMPILE_OPTIONS /std:c++20)
    else()
        set_property(SOURCE llama-parquet.cpp APPEND PROPERTY COMPILE_OPTIONS -std=c++20)
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
