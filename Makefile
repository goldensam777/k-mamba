# Makefile — k-mamba (zero dependency, optional CUDA)
# Build: make
# Build with CUDA: make (auto-detected if nvcc available)
# Force CPU only: make CPU_ONLY=1
# Clean: make clean
# Test: make test

CC = gcc
CFLAGS = -O3 -mavx2 -Wall -Wextra -I./include
LDFLAGS = -lm

# ═══════════════════════════════════════════════════════════════
# CUDA Auto-Detection
# ═══════════════════════════════════════════════════════════════
NVCC := $(shell which nvcc 2>/dev/null)
CUDA_AVAILABLE := $(if $(NVCC),1,0)

# Allow forcing CPU-only build
ifdef CPU_ONLY
CUDA_AVAILABLE := 0
endif

# CUDA settings if available
ifeq ($(CUDA_AVAILABLE),1)
CUDA_HOME ?= $(dir $(NVCC))..
CUDA_FLAGS = -O3 -arch=sm_70 -I./include -I$(CUDA_HOME)/include
CUDA_LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CFLAGS += -DKMAMBA_BUILD_CUDA

CUDA_SRCS = cuda/scan1d.cu \
            cuda/scan1d_backward.cu \
            cuda/scan_nd.cu \
            cuda/convnd.cu \
            cuda/mamba_scan.cu \
            cuda/mamba_block.cu

CUDA_OBJS = $(CUDA_SRCS:.cu=.cu.o)
CUDA_TARGET = libkmamba_cuda.a
else
CUDA_OBJS =
CUDA_TARGET =
endif

# ═══════════════════════════════════════════════════════════════
# Source files
# ═══════════════════════════════════════════════════════════════
SRCS = src/kmamba.c \
       src/mamba_block.c \
       src/kmamba_cuda_utils.c \
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

# ═══════════════════════════════════════════════════════════════
# Build targets
# ═══════════════════════════════════════════════════════════════

# Default target
all: check_cuda $(TARGET)

# CUDA availability check
check_cuda:
ifeq ($(CUDA_AVAILABLE),1)
	@echo "✓ CUDA detected: $(NVCC)"
	@echo "  Building with GPU support..."
else
	@echo "✗ CUDA not detected (nvcc not found)"
	@echo "  Building CPU-only version..."
	@echo "  To force CPU-only: make CPU_ONLY=1"
endif

# Static library
$(TARGET): $(OBJS) $(ASM_OBJS) $(CUDA_OBJS)
	ar rcs $@ $^
ifeq ($(CUDA_AVAILABLE),1)
	@echo ""
	@echo "=== libkmamba.a built WITH CUDA support ==="
	@echo "GPU will be used automatically if available at runtime"
else
	@echo ""
	@echo "=== libkmamba.a built CPU-only ==="
endif

# C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ASM files (NASM)
%.o: %.asm
	nasm -f elf64 -O3 -I./include $< -o $@

# CUDA files (only if CUDA available)
ifeq ($(CUDA_AVAILABLE),1)
%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@
endif

# ═══════════════════════════════════════════════════════════════
# Maintenance
# ═══════════════════════════════════════════════════════════════

clean:
	rm -f $(OBJS) $(ASM_OBJS) $(CUDA_OBJS) $(TARGET)
	rm -f cuda/*.cu.o
	rm -f src/*.o kernels/*.o cpu/*.o

test: $(TARGET)
	@echo "Build successful: $(TARGET)"
	@echo "Library contents:"
	ar -t $(TARGET)
	@echo ""
ifeq ($(CUDA_AVAILABLE),1)
	@echo "CUDA objects in library:"
	ar -t $(TARGET) | grep '\.cu\.o' || true
endif

.PHONY: all clean test check_cuda
