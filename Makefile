# Makefile — k-mamba (zero dependency)
# Build: make
# Clean: make clean
# Test: make test

CC = gcc
CFLAGS = -O3 -mavx2 -Wall -Wextra -I./include
LDFLAGS = -lm

# Source files
SRCS = src/kmamba.c \
       src/mamba_block.c \
       src/km_topology.c \
       src/wavefront_nd.c \
       src/wavefront_plan.c \
       src/scan_nd.c \
       src/convnd.c \
       kernels/gemm_f32.c \
       kernels/activations_f32.c \
       kernels/elementwise_f32.c \
       kernels/optimizer_f32.c \
       kernels/init_f32.c

# ASM files
ASM_SRCS = cpu/scan1d.asm \
           cpu/scan2d.asm

# Object files
OBJS = $(SRCS:.c=.o)
ASM_OBJS = $(ASM_SRCS:.asm=.o)

# Target library
TARGET = libkmamba.a

# Default target
all: $(TARGET)

# Static library
$(TARGET): $(OBJS) $(ASM_OBJS)
	ar rcs $@ $^

# C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ASM files (NASM)
%.o: %.asm
	nasm -f elf64 -O3 -I./include $< -o $@

# Clean
clean:
	rm -f $(OBJS) $(ASM_OBJS) $(TARGET)

# Test build (basic test)
test: $(TARGET)
	@echo "Build successful: $(TARGET)"
	@echo "Library contents:"
	ar -t $(TARGET)

.PHONY: all clean test
