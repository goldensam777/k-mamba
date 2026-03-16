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

# Tests unitaires pour les kernels optimatrix
add_executable(test_optimatrix_kernels unit/test_optimatrix_kernels.c)

# Linker avec optimatrix
target_link_libraries(test_optimatrix_kernels PRIVATE optimatrix::optimatrix m)

# Flags de compilation pour les tests
target_compile_options(test_optimatrix_kernels PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Tests Benchmarks
add_executable(test_benchmarks integration/test_benchmarks.c)
target_link_libraries(test_benchmarks PRIVATE 
    optimatrix::optimatrix
    k-mamba-cpu
    m
)
target_compile_options(test_benchmarks PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Tests ConvND (version réécrite)
add_executable(test_convnd integration/test_convnd.c)
target_link_libraries(test_convnd PRIVATE m)
target_compile_options(test_convnd PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Tests KMamba inférence (working version)
add_executable(test_kmamba_inference_working end_to_end/test_kmamba_inference_working.c)
target_link_libraries(test_kmamba_inference_working PRIVATE m)
target_compile_options(test_kmamba_inference_working PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Tests KMamba inférence (version réécrite)
add_executable(test_kmamba_inference end_to_end/test_kmamba_inference.c)
target_link_libraries(test_kmamba_inference PRIVATE m)
target_compile_options(test_kmamba_inference PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Tests KMamba training (version réécrite)
add_executable(test_kmamba_training end_to_end/test_kmamba_training.c)
target_link_libraries(test_kmamba_training PRIVATE m)
target_compile_options(test_kmamba_training PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Tests gradient utils
add_executable(test_gradient_utils unit/test_gradient_utils.c)
target_link_libraries(test_gradient_utils PRIVATE optimatrix::optimatrix m)
target_compile_options(test_gradient_utils PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Tests optimizers (nouveau)
add_executable(test_optimizers_new unit/test_optimizers.c)
target_link_libraries(test_optimizers_new PRIVATE m)
target_compile_options(test_optimizers_new PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Tests simples
add_executable(test_simple unit/test_simple.c)
target_link_libraries(test_simple PRIVATE m)

# Tests benchmarks (version réécrite)
add_executable(test_benchmarks_new integration/test_benchmarks.c)
target_link_libraries(test_benchmarks_new PRIVATE m)
target_compile_options(test_benchmarks_new PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Ajouter aux tests CTest
add_test(NAME optimatrix_kernels_unit COMMAND test_optimatrix_kernels)
add_test(NAME utilitaires_unit COMMAND test_utilitaires)
add_test(NAME kmamba_inference_working COMMAND test_kmamba_inference_working)
add_test(NAME convnd_integration COMMAND test_convnd)
add_test(NAME kmamba_inference_e2e COMMAND test_kmamba_inference)
add_test(NAME kmamba_training_e2e COMMAND test_kmamba_training)
add_test(NAME gradient_utils_unit COMMAND test_gradient_utils)
add_test(NAME optimizers_unit COMMAND test_optimizers_new)
add_test(NAME simple_test COMMAND test_simple)

# Activer les tests
enable_testing()
add_test(NAME OptimizersTest COMMAND test_optimizers_new)

# Tests GPU simples
if(KMAMBA_BUILD_CUDA)
    add_executable(test_gpu_simple gpu/test_gpu_simple.cu)
    target_link_libraries(test_gpu_simple PRIVATE 
        cuda
        m
    )
    target_compile_options(test_gpu_simple PRIVATE
        -O3 -mavx2 -Wall -Wextra -g
    )
    
    # Ajouter aux tests CTest
    add_test(NAME gpu_simple_test COMMAND test_gpu_simple)
    
    enable_testing()
    add_test(NAME GpuSimpleTest COMMAND test_gpu_simple)
endif()
