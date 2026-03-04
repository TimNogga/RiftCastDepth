cmake_minimum_required(VERSION 3.27)

project(maskcompression LANGUAGES C CXX CUDA VERSION 1.0.0)

option(MASKCOMPRESSION_BUILD_BINDINGS "Build the Python bindings" ON)

find_package(charonload)

if(charonload_FOUND)
    charonload_add_torch_library(maskcompression_cpp STATIC)
    add_library(maskcompression::maskcompression_cpp ALIAS maskcompression_cpp)

    target_sources(maskcompression_cpp PRIVATE src/decompress.cu
                                               src/compress.cpp)

    target_include_directories(maskcompression_cpp PUBLIC include)

    if(MASKCOMPRESSION_BUILD_BINDINGS)
        charonload_add_torch_library(${TORCH_EXTENSION_NAME} MODULE)

        target_sources(${TORCH_EXTENSION_NAME} PRIVATE python/maskcompression.cpp)

        target_link_libraries(${TORCH_EXTENSION_NAME} PRIVATE maskcompression_cpp)
    endif()
endif()