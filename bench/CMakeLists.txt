cmake_minimum_required(VERSION 3.18)
project(k-mamba-bench)

if(KMAMBA_BUILD_CUDA)
    enable_language(CUDA)

    # bench_gpu.cu compiled as a CUDA object library
    add_library(bench_gpu_lib STATIC bench_gpu.cu)
    target_include_directories(bench_gpu_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include
    )
    target_link_libraries(bench_gpu_lib PRIVATE
        k-mamba-cuda
        cuda cudart
    )
    set_target_properties(bench_gpu_lib PROPERTIES
        CUDA_ARCHITECTURES "75"
    )
    target_compile_options(bench_gpu_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    )
endif()

add_executable(bench_paper bench_paper.c)

target_include_directories(bench_paper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include
)

target_link_libraries(bench_paper PRIVATE
    k-mamba
    k-mamba-cpu
    optimatrix::optimatrix
    m
)

if(KMAMBA_BUILD_CUDA)
    target_compile_definitions(bench_paper PRIVATE KMAMBA_HAS_CUDA)
    target_link_libraries(bench_paper PRIVATE bench_gpu_lib cuda cudart)
endif()

target_link_options(bench_paper PRIVATE -no-pie)
target_compile_options(bench_paper PRIVATE -O3 -mavx2)
