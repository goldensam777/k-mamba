cmake_minimum_required(VERSION 3.18)
project(k-mamba-cpu VERSION 1.0.0 LANGUAGES C ASM_NASM)

# NASM configuration
set(CMAKE_ASM_NASM_COMPILE_OBJECT "<CMAKE_ASM_NASM_COMPILER> <INCLUDES> <FLAGS> -o <OBJECT> <SOURCE>")
set(CMAKE_ASM_NASM_FLAGS "-f elf64")

# Mamba-specific CPU kernels
set(CPU_ASM_SRCS
    scan1d.asm
    scan2d.asm
    # scan1d_backward_m1_shared_bc.asm  # Temporarily disabled due to ASM syntax error
    # scan1d_backward_m1_shared_bc_simple.asm  # Temporarily disabled
)

set(CPU_C_SRCS
    # scan1d_backward.c  # Temporarily disabled due to header issues
    scan1d_backward_m.c
    mamba_scan.c
)

# Compile ASM files
foreach(ASM_SRC ${CPU_ASM_SRCS})
    get_filename_component(FILENAME ${ASM_SRC} NAME_WE)
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/cpu_${FILENAME}.o")
    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${CMAKE_ASM_NASM_COMPILER} ${CMAKE_ASM_NASM_FLAGS} 
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include
                ${CMAKE_CURRENT_SOURCE_DIR}/${ASM_SRC} -o ${OBJ_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ASM_SRC}
        COMMENT "Assembling ${ASM_SRC}"
    )
    list(APPEND CPU_OBJS ${OBJ_FILE})
endforeach()

# Compile C files
add_library(k-mamba-cpu STATIC ${CPU_C_SRCS} ${CPU_OBJS})

target_include_directories(k-mamba-cpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include>
)

target_compile_options(k-mamba-cpu PRIVATE -O3 -mavx2)

# Link with math library
target_link_libraries(k-mamba-cpu PRIVATE m)
