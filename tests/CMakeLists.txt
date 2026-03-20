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
    k-mamba
    k-mamba-cpu
    optimatrix::optimatrix
    m
)


# Tests unitaires pour les kernels optimatrix
add_executable(test_optimatrix_kernels unit/test_optimatrix_kernels.c)

# Linker avec optimatrix
target_link_libraries(test_optimatrix_kernels PRIVATE optimatrix::optimatrix m)

# ASM utilise des relocations 32-bit : désactiver PIE pour les exécutables de test
target_link_options(test_optimizers       PRIVATE -no-pie)
target_link_options(test_optimatrix_kernels PRIVATE -no-pie)

# Flags de compilation pour les tests
target_compile_options(test_optimatrix_kernels PRIVATE
    -O3 -mavx2 -Wall -Wextra -g
)

# Test backward ASM vs C
add_executable(test_scan1d_backward_asm unit/test_scan1d_backward_asm.c)
target_link_libraries(test_scan1d_backward_asm PRIVATE k-mamba k-mamba-cpu optimatrix::optimatrix m)
target_link_options(test_scan1d_backward_asm PRIVATE -no-pie)
target_compile_options(test_scan1d_backward_asm PRIVATE -O3 -mavx2 -Wall -Wextra -g)

# Test scan1d forward : ASM vs référence C
add_executable(test_scan1d unit/test_scan1d.c)
target_link_libraries(test_scan1d PRIVATE k-mamba k-mamba-cpu optimatrix::optimatrix m)
target_link_options(test_scan1d PRIVATE -no-pie)
target_compile_options(test_scan1d PRIVATE -O3 -mavx2 -Wall -Wextra -g)

# Test wavefront ND : générateur topologique borné
add_executable(test_wavefront_nd unit/test_wavefront_nd.c)
target_link_libraries(test_wavefront_nd PRIVATE k-mamba k-mamba-cpu optimatrix::optimatrix m)
target_link_options(test_wavefront_nd PRIVATE -no-pie)
target_compile_options(test_wavefront_nd PRIVATE -O3 -mavx2 -Wall -Wextra -g)

# Test scan ND : référence + dispatch spécialisé 1D/2D
add_executable(test_scan_nd unit/test_scan_nd.c)
target_link_libraries(test_scan_nd PRIVATE k-mamba k-mamba-cpu optimatrix::optimatrix m)
target_link_options(test_scan_nd PRIVATE -no-pie)
target_compile_options(test_scan_nd PRIVATE -O3 -mavx2 -Wall -Wextra -g)

# Test scan1d backward : gradient check numérique
add_executable(test_scan1d_backward unit/test_scan1d_backward.c)
target_link_libraries(test_scan1d_backward PRIVATE k-mamba k-mamba-cpu optimatrix::optimatrix m)
target_link_options(test_scan1d_backward PRIVATE -no-pie)
target_compile_options(test_scan1d_backward PRIVATE -O3 -mavx2 -Wall -Wextra -g)

# Test end-to-end KMamba
add_executable(test_kmamba_e2e integration/test_kmamba_e2e.c)
target_link_libraries(test_kmamba_e2e PRIVATE k-mamba k-mamba-cpu optimatrix::optimatrix m)
target_link_options(test_kmamba_e2e PRIVATE -no-pie)
target_compile_options(test_kmamba_e2e PRIVATE -O3 -mavx2 -Wall -Wextra -g)

# Ajouter aux tests CTest
add_test(NAME optimatrix_kernels_unit COMMAND test_optimatrix_kernels)
add_test(NAME Scan1DBackwardASM COMMAND test_scan1d_backward_asm)
add_test(NAME Scan1DForwardASM COMMAND test_scan1d)
add_test(NAME WavefrontNDTest COMMAND test_wavefront_nd)
add_test(NAME ScanNDTest COMMAND test_scan_nd)
add_test(NAME Scan1DBackwardGrad COMMAND test_scan1d_backward)
add_test(NAME KMambaE2E COMMAND test_kmamba_e2e)

# Activer les tests
enable_testing()
add_test(NAME OptimizersTest COMMAND test_optimizers)

if(KMAMBA_BUILD_CUDA)
    # Test scan1d CUDA vs reference CPU
    add_executable(test_cuda_scan test_cuda_scan.cu)

    target_include_directories(test_cuda_scan PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include
    )

    target_link_libraries(test_cuda_scan PRIVATE
        k-mamba-cuda
        cuda cudart
    )

    set_target_properties(test_cuda_scan PROPERTIES
        CUDA_ARCHITECTURES "75"
    )

    target_compile_options(test_cuda_scan PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )

    add_test(NAME CudaScan1DTest COMMAND test_cuda_scan)

    # Test scan_nd CUDA vs reference CPU
    add_executable(test_cuda_scan_nd test_cuda_scan_nd.cu)

    target_include_directories(test_cuda_scan_nd PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include
    )

    target_link_libraries(test_cuda_scan_nd PRIVATE
        k-mamba
        k-mamba-cuda
        cuda cudart
    )
    target_link_options(test_cuda_scan_nd PRIVATE -no-pie)

    set_target_properties(test_cuda_scan_nd PROPERTIES
        CUDA_ARCHITECTURES "75"
    )

    target_compile_options(test_cuda_scan_nd PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )

    add_test(NAME CudaScanNDTest COMMAND test_cuda_scan_nd)
endif()
