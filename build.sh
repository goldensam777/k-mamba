#!/bin/bash
# build.sh — k-mamba zero-dependency build script
# Usage: ./build.sh [clean]

set -e

CC=${CC:-gcc}
CFLAGS="-O3 -mavx2 -Wall -Wextra -I./include"
LDFLAGS="-lm"
TARGET="libkmamba.a"

echo "=== k-mamba build ==="

# Clean if requested
if [ "$1" == "clean" ]; then
    echo "Cleaning..."
    rm -f src/*.o kernels/*.o cpu/*.o $(TARGET)
    echo "Clean done."
    exit 0
fi

# Compile C sources
echo "Compiling C sources..."
for src in src/kmamba.c src/mamba_block.c src/km_topology.c src/wavefront_nd.c \
           src/wavefront_plan.c src/scan_nd.c src/convnd.c \
           kernels/gemm_f32.c kernels/activations_f32.c \
           kernels/elementwise_f32.c kernels/optimizer_f32.c kernels/init_f32.c; do
    obj="${src%.c}.o"
    echo "  $src -> $obj"
    $CC $CFLAGS -c $src -o $obj
done

# Compile ASM sources
echo "Compiling ASM sources..."
for src in cpu/scan1d.asm cpu/scan2d.asm; do
    obj="${src%.asm}.o"
    echo "  $src -> $obj"
    nasm -f elf64 -O3 $src -o $obj
done

# Create static library
echo "Creating static library..."
ar rcs $TARGET \
    src/*.o kernels/*.o cpu/*.o

echo "=== Build complete: $TARGET ==="
echo "Library contents:"
ar -t $TARGET
