cmake_minimum_required(VERSION 3.10)
project(LibTRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Werror -Wno-unknown-pragmas -Wno-deprecated-declarations")

file(GLOB source_files
        "TRTLogger/*.h"
        "TRTLogger/*.cpp"
        "TRTRuntime/*.h"
        "TRTRuntime/*cpp"
        "TRTCudaEngine/*.h"
        "TRTCudaEngine/*.cpp"
        "TRTContext/*.h"
        "TRTContext/*.cpp"
        "TRTUffParser/*.h"
        "TRTUffParser/*.cpp"
        "TRTDims/*.h"
        "TRTDims/*.cpp"
        "TRTBuilder/*.h"
        "TRTBuilder/*.cpp"
        "TRTNetworkDefinition/*.h"
        "TRTNetworkDefinition/*.cpp"
        "TRTHostMemory/*.h"
        "TRTHostMemory/*.cpp"
        "*.h"
        )

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

#add_executable(trt-sys-test ${source_files} main.c)
add_library(trt-sys STATIC ${source_files})
target_link_libraries(trt-sys PRIVATE nvinfer ${CUDART_LIBRARY})
#target_link_libraries(trt-sys-test PRIVATE nvinfer ${CUDART_LIBRARY})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

install(TARGETS trt-sys DESTINATION .)
