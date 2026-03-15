cmake_minimum_required(VERSION 3.18)
project(k-mamba-tests)

# Inclure les headers de k-mamba et optimatrix
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include
)

# Test des optimiseurs CPU
add_executable(test_optimizers test_optimizers.c)
target_link_libraries(test_optimizers 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/mamba_block.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/src/optimizer_utils.c
    m
)

# Test des optimiseurs CUDA
if(KMAMBA_BUILD_CUDA)
    enable_language(CUDA)
    add_executable(test_cuda_optimizers test_cuda_optimizers.cu)
    
    # Inclure les sources CUDA
    target_sources(test_cuda_optimizers PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/src/optimizer_utils.cu
    )
    
    target_link_libraries(test_cuda_optimizers PRIVATE cuda cudart)
    
    # Options de compilation CUDA
    target_compile_options(test_cuda_optimizers PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        -arch=sm_70
    )
endif()

# Activer les tests
enable_testing()
add_test(NAME OptimizersTest COMMAND test_optimizers)

if(KMAMBA_BUILD_CUDA)
    add_test(NAME CudaOptimizersTest COMMAND test_cuda_optimizers)
endif()
