# Makefile maître — k-mamba
# Structure inspirée de optimus/cpu/Makefile
#
# Usage:
#   make all              # Tout compiler (lib + modèles)
#   make lib              # Juste la bibliothèque
#   make cpu_lm_model     # Modèle CPU 500K
#   make cuda_lm_model    # Modèle GPU 500M
#   make hybrid_lm_model  # Modèle Hybrid 1.5M
#   make all_models       # Tous les modèles
#   make tests            # Tests unitaires
#   make clean            # Nettoyage

.PHONY: all lib models cpu_lm_model cuda_lm_model hybrid_lm_model all_models tests clean distclean

# ═══════════════════════════════════════════════════════════════
# Compilateurs et flags
# ═══════════════════════════════════════════════════════════════
CC = gcc
CFLAGS = -O3 -mavx2 -Wall -Wextra -I./include -fopenmp
LDFLAGS = -lm -lgomp

# ═══════════════════════════════════════════════════════════════
# Rust Tokenizer
# ═══════════════════════════════════════════════════════════════
RUST_DIR = tokenizer_rs
RUST_LIB = $(RUST_DIR)/target/release/libkmamba_tokenizer.a
RUST_LDFLAGS = -lrt -ldl -lpthread

CARGO := $(shell which cargo 2>/dev/null)
RUST_AVAILABLE := $(if $(CARGO),1,0)

# ═══════════════════════════════════════════════════════════════
# CUDA Auto-Detection
# ═══════════════════════════════════════════════════════════════
NVCC := $(shell which nvcc 2>/dev/null)
CUDA_AVAILABLE := $(if $(NVCC),1,0)

ifdef CPU_ONLY
CUDA_AVAILABLE := 0
endif

ifeq ($(CUDA_AVAILABLE),1)
CUDA_HOME ?= $(dir $(NVCC))..
CUDA_FLAGS = -O3 -arch=sm_70 -I./include -I$(CUDA_HOME)/include -DKMAMBA_BUILD_CUDA
CUDA_LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CFLAGS += -DKMAMBA_BUILD_CUDA
endif

# ═══════════════════════════════════════════════════════════════
# Fichiers source
# ═══════════════════════════════════════════════════════════════
SRCS = src/kmamba.c \
       src/mamba_block.c \
       src/kmamba_cuda_utils.c \
       src/kmamba_mixed_precision.c \
       src/kmamba_checkpoint.c \
       src/kmamba_distributed.c \
       src/km_topology.c \
       src/wavefront_nd.c \
       src/wavefront_plan.c \
       src/scan_nd.c \
       src/convnd.c \
       src/km_memory_pool.c \
       kernels/gemm_f32.c \
       kernels/activations_f32.c \
       kernels/elementwise_f32.c \
       kernels/optimizer_f32.c \
       kernels/init_f32.c

ASM_SRCS = cpu/scan1d.asm \
           cpu/scan2d.asm

CUDA_SRCS = cuda/scan1d.cu \
            cuda/scan1d_backward.cu \
            cuda/scan_nd.cu \
            cuda/convnd.cu \
            cuda/mamba_scan.cu \
            cuda/mamba_block.cu \
            cuda/kmamba_cuda_utils.cu \
            cuda/kmamba_mixed_precision.cu \
            cuda/kmamba_checkpoint.cu \
            cuda/kmamba_distributed.cu

# ═══════════════════════════════════════════════════════════════
# Objets et cibles
# ═══════════════════════════════════════════════════════════════
OBJS = $(SRCS:.c=.o)
ASM_OBJS = $(ASM_SRCS:.asm=.o)
CUDA_OBJS = $(patsubst %.cu,cuda/%.o,$(notdir $(CUDA_SRCS)))

TARGET = libkmamba.a

MODEL_CPU = models/kmamba_cpu
MODEL_CUDA = models/kmamba_cuda
MODEL_HYBRID = models/kmamba_hybrid

# ═══════════════════════════════════════════════════════════════
# Cibles principales
# ═══════════════════════════════════════════════════════════════

# Tout compiler
all: lib all_models

# Juste la bibliothèque
lib: check_cuda check_rust $(RUST_LIB) $(TARGET)
	@echo ""
	@echo "=== libkmamba.a prête ==="

# Tous les modèles
models: all_models

# Modèle CPU uniquement
cpu_lm_model: lib $(MODEL_CPU)
	@echo "✓ Modèle CPU prêt: $(MODEL_CPU)"

# Modèle CUDA uniquement (si dispo)
cuda_lm_model: lib $(MODEL_CUDA)
ifeq ($(CUDA_AVAILABLE),1)
	@echo "✓ Modèle CUDA prêt: $(MODEL_CUDA)"
else
	@echo "✗ CUDA non disponible — modèle CUDA ignoré"
endif

# Modèle Hybrid uniquement (si dispo)
hybrid_lm_model: lib $(MODEL_HYBRID)
ifeq ($(CUDA_AVAILABLE),1)
	@echo "✓ Modèle Hybrid prêt: $(MODEL_HYBRID)"
else
	@echo "✗ CUDA non disponible — modèle Hybrid ignoré"
endif

# Tous les modèles (selon disponibilité CUDA)
all_models: cpu_lm_model
ifeq ($(CUDA_AVAILABLE),1)
	@echo "Compilation modèles CUDA..."
	@$(MAKE) $(MODEL_CUDA) $(MODEL_HYBRID)
	@echo "✓ Tous les modèles sont prêts"
endif

# Tests
tests: lib
	@echo "=== Compilation des tests ==="
	@$(MAKE) test-mamba3
ifeq ($(CUDA_AVAILABLE),1)
	@$(MAKE) test-mamba3-gpu
endif

# ═══════════════════════════════════════════════════════════════
# Compilation de la lib
# ═══════════════════════════════════════════════════════════════

$(TARGET): $(OBJS) $(ASM_OBJS) $(CUDA_OBJS)
	ar rcs $@ $^

# C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ASM files
%.o: %.asm
	nasm -f elf64 -O3 -I./include $< -o $@

# CUDA files
ifeq ($(CUDA_AVAILABLE),1)
cuda/%.o: cuda/%.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@
endif

# Rust tokenizer
$(RUST_LIB):
ifeq ($(RUST_AVAILABLE),1)
	cd $(RUST_DIR) && cargo build --release
	@echo "✓ Tokenizer Rust compilé"
else
	@echo "✗ Rust indisponible — tokenizer ignoré"
endif

# ═══════════════════════════════════════════════════════════════
# Compilation des modèles
# ═══════════════════════════════════════════════════════════════

# Crée le répertoire models
models_dir:
	@mkdir -p models

# Lien des libs
MODEL_LDFLAGS = $(TARGET) $(LDFLAGS)
ifeq ($(RUST_AVAILABLE),1)
MODEL_LDFLAGS += $(RUST_LIB) $(RUST_LDFLAGS)
endif
ifeq ($(CUDA_AVAILABLE),1)
MODEL_LDFLAGS += $(CUDA_LDFLAGS)
endif

# Modèle CPU
$(MODEL_CPU): models/kmamba_cpu.c $(TARGET) $(RUST_LIB) | models_dir
	$(CC) $(CFLAGS) -o $@ $< $(MODEL_LDFLAGS)
	@echo "Built: $@ (CPU 500K params, BPE 32K)"

# Modèle CUDA (fichier .cu compilé avec nvcc)
$(MODEL_CUDA): models/kmamba_cuda.cu $(TARGET) $(RUST_LIB) | models_dir
ifeq ($(CUDA_AVAILABLE),1)
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(TARGET) $(CUDA_LDFLAGS) $(RUST_LIB) $(RUST_LDFLAGS) -Xcompiler "$(CFLAGS)"
	@echo "Built: $@ (CUDA 500M params, BPE 32K)"
endif

# Modèle Hybrid
$(MODEL_HYBRID): models/kmamba_hybrid.c $(TARGET) $(RUST_LIB) | models_dir
ifeq ($(CUDA_AVAILABLE),1)
	$(CC) $(CFLAGS) -o $@ $< $(MODEL_LDFLAGS)
	@echo "Built: $@ (Hybrid 1.5M params, BPE 32K)"
endif

# ═══════════════════════════════════════════════════════════════
# Vérifications
# ═══════════════════════════════════════════════════════════════

check_cuda:
ifeq ($(CUDA_AVAILABLE),1)
	@echo "✓ CUDA: $(NVCC)"
else
	@echo "✗ CUDA non détecté"
endif

check_rust:
ifeq ($(RUST_AVAILABLE),1)
	@echo "✓ Rust/Cargo disponible"
else
	@echo "✗ Rust indisponible"
endif

# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

test-mamba3: $(TARGET) tests/unit/test_mamba3_forward.c
ifeq ($(CUDA_AVAILABLE),1)
	$(CC) $(CFLAGS) -o test_mamba3 tests/unit/test_mamba3_forward.c $(TARGET) $(LDFLAGS) $(CUDA_LDFLAGS)
else
	$(CC) $(CFLAGS) -o test_mamba3 tests/unit/test_mamba3_forward.c $(TARGET) $(LDFLAGS)
endif
	./test_mamba3

test-mamba3-gpu: $(TARGET) tests/unit/test_mamba3_gpu.cu
ifeq ($(CUDA_AVAILABLE),1)
	$(NVCC) -O3 -arch=sm_70 -I./include -I$(CUDA_HOME)/include -o test_mamba3_gpu tests/unit/test_mamba3_gpu.cu $(TARGET) -L$(CUDA_HOME)/lib64 -lcudart -lcublas
	./test_mamba3_gpu
else
	@echo "SKIP: test GPU nécessite CUDA"
endif

# ═══════════════════════════════════════════════════════════════
# Nettoyage
# ═══════════════════════════════════════════════════════════════

clean:
	rm -f $(OBJS) $(ASM_OBJS)
	rm -f src/*.o kernels/*.o cpu/*.o
	rm -f test_mamba3 test_mamba3_gpu
ifeq ($(CUDA_AVAILABLE),1)
	rm -f cuda/*.cu.o
endif
ifeq ($(RUST_AVAILABLE),1)
	cd $(RUST_DIR) && cargo clean 2>/dev/null || true
endif

distclean: clean
	rm -f $(TARGET) $(RUST_LIB)
	rm -rf models/
	rm -f examples/train_500k examples/train_500m examples/train_1_5m examples/chat

# ═══════════════════════════════════════════════════════════════
# Help
# ═══════════════════════════════════════════════════════════════

help:
	@echo "k-mamba Makefile maître"
	@echo ""
	@echo "Cibles principales:"
	@echo "  make all              - Tout compiler (lib + modèles)"
	@echo "  make lib              - Juste la bibliothèque libkmamba.a"
	@echo "  make cpu_lm_model     - Modèle CPU 500K params"
	@echo "  make cuda_lm_model    - Modèle GPU 500M params (si CUDA)"
	@echo "  make hybrid_lm_model  - Modèle Hybrid 1.5M params (si CUDA)"
	@echo "  make all_models       - Tous les modèles disponibles"
	@echo "  make tests            - Tests unitaires"
	@echo "  make clean            - Nettoyage"
	@echo "  make distclean        - Nettoyage complet"
	@echo "  make help             - Cette aide"
	@echo ""
	@echo "Variables:"
	@echo "  CPU_ONLY=1            - Forcer compilation CPU sans CUDA"
