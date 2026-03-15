cmake_minimum_required(VERSION 3.18)
project(k-mamba VERSION 0.1.0 LANGUAGES C ASM_NASM)

# Options
option(KMAMBA_BUILD_TESTS "Build tests" OFF)
option(KMAMBA_BUILD_EXAMPLES "Build examples" OFF)
option(KMAMBA_BUILD_CPU "Build CPU backend" ON)
option(KMAMBA_BUILD_CUDA "Build CUDA backend" OFF)

# NASM flags
set(CMAKE_ASM_NASM_COMPILE_OBJECT "<CMAKE_ASM_NASM_COMPILER> <INCLUDES> <FLAGS> -o <OBJECT> <SOURCE>")
set(CMAKE_ASM_NASM_FLAGS "-f elf64 -O3")

# Find optimatrix submodule
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/optimatrix/CMakeLists.txt)
    message(FATAL_ERROR "optimatrix submodule not found. Run: git submodule update --init --recursive")
endif()

add_subdirectory(optimatrix)

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
)

add_library(k-mamba STATIC ${KMAMBA_C_SRCS})

add_library(k-mamba::k-mamba ALIAS k-mamba)

target_include_directories(k-mamba
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/optimatrix/include
)

target_link_libraries(k-mamba
    PUBLIC optimatrix::optimatrix
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

# Installation
include(GNUInstallDirs)

install(TARGETS k-mamba
    EXPORT k-mamba-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(KMAMBA_BUILD_CPU)
    install(TARGETS k-mamba-cpu
        EXPORT k-mamba-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(KMAMBA_BUILD_CUDA)
    install(TARGETS k-mamba-cuda
        EXPORT k-mamba-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT k-mamba-targets
    FILE k-mambaTargets.cmake
    NAMESPACE k-mamba::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/k-mamba
)

# Package config
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/k-mambaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/k-mambaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/k-mamba
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/k-mambaConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/k-mambaConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/k-mambaConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/k-mamba
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
