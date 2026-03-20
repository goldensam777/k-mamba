cmake_minimum_required(VERSION 3.18)
project(k-mamba-cuda VERSION 1.0.0 LANGUAGES CUDA)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

# Mamba-specific CUDA kernels
set(CUDA_SRCS
    scan1d.cu
    scan_nd.cu
    scan1d_backward.cu
    mamba_scan.cu
    mamba_block.cu
)

add_library(k-mamba-cuda STATIC ${CUDA_SRCS})

target_include_directories(k-mamba-cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include>
)

target_compile_options(k-mamba-cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>
)

# Link CUDA libraries
target_link_libraries(k-mamba-cuda
    PUBLIC  optimatrix-cuda
    PRIVATE cuda cudart cublas
)
