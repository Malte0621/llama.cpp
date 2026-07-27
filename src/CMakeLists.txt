llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
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
            llama-kv-cache-dsv4.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
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
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

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
