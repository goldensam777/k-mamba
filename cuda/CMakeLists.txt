cmake_minimum_required(VERSION 3.18)
project(k-mamba-cuda VERSION 1.0.0 LANGUAGES CUDA)

# Find CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

# Mamba-specific CUDA kernels
set(CUDA_SRCS
    scan1d.cu
    scan1d_backward.cu
    mamba_scan.cu
)

# Compile CUDA kernels
add_library(k-mamba-cuda STATIC ${CUDA_SRCS})

target_include_directories(k-mamba-cuda
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include
)

# CUDA compile options
target_compile_options(k-mamba-cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    -arch=sm_70
)

# Link CUDA libraries
target_link_libraries(k-mamba-cuda PRIVATE cuda cudart)
