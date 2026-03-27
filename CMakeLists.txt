cmake_minimum_required(VERSION 3.18)
project(k-mamba VERSION 0.1.0 LANGUAGES C ASM_NASM)

find_package(OpenMP)
find_package(OpenBLAS REQUIRED)

# Options
option(KMAMBA_BUILD_TESTS "Build tests" OFF)
option(KMAMBA_BUILD_EXAMPLES "Build examples" OFF)
option(KMAMBA_BUILD_CPU "Build CPU backend" ON)
option(KMAMBA_BUILD_CUDA "Build CUDA backend" OFF)

# Add optimatrix when CUDA or tests need its exported targets
if(KMAMBA_BUILD_CUDA OR KMAMBA_BUILD_TESTS)
    add_subdirectory(optimatrix)
endif()

# NASM flags
set(CMAKE_ASM_NASM_COMPILE_OBJECT "<CMAKE_ASM_NASM_COMPILER> <INCLUDES> <FLAGS> -o <OBJECT> <SOURCE>")
set(CMAKE_ASM_NASM_FLAGS "-f elf64 -O3")

# K-Mamba specific backends
if(KMAMBA_BUILD_CPU)
    add_subdirectory(cpu)
endif()

if(KMAMBA_BUILD_CUDA)
    enable_language(CUDA)
    add_subdirectory(cuda)
endif()

# Library target - Mamba orchestration only
set(KMAMBA_C_SRCS
    src/kmamba.c
    src/mamba_block.c
    src/convnd.c
    src/scan_nd.c
    src/wavefront_plan.c
    src/wavefront_nd.c
    src/openblas_utils.c
)

add_library(k-mamba STATIC ${KMAMBA_C_SRCS})

add_library(k-mamba::k-mamba ALIAS k-mamba)

target_include_directories(k-mamba
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(k-mamba
    PUBLIC ${OpenBLAS_LIBRARIES}
)

target_include_directories(k-mamba
    PUBLIC ${OpenBLAS_INCLUDE_DIRS}
)

# Link K-Mamba specific backends
if(KMAMBA_BUILD_CPU)
    target_link_libraries(k-mamba PUBLIC k-mamba-cpu)
endif()

if(KMAMBA_BUILD_CUDA)
    target_link_libraries(k-mamba PUBLIC k-mamba-cuda)
endif()

target_compile_options(k-mamba PRIVATE
    -O3 -mavx2 -Wall -Wextra -Wpedantic
)

if(OpenMP_C_FOUND)
    target_link_libraries(k-mamba PUBLIC OpenMP::OpenMP_C)
endif()

if(KMAMBA_BUILD_CUDA)
    target_compile_definitions(k-mamba PRIVATE KMAMBA_BUILD_CUDA=1)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS k-mamba
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(KMAMBA_BUILD_CPU)
    install(TARGETS k-mamba-cpu
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(KMAMBA_BUILD_CUDA)
    install(TARGETS k-mamba-cuda
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Tests
if(KMAMBA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(KMAMBA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Benchmarks (paper)
option(KMAMBA_BUILD_BENCH "Build paper benchmarks" OFF)
if(KMAMBA_BUILD_BENCH)
    add_subdirectory(bench)
endif()
