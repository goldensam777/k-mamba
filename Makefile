CC      = gcc
NASM    = nasm
NVCC    = nvcc
# enable OpenMP when available for parallel selective_scan
CFLAGS  = -Wall -Wextra -O2 -std=c99 -mavx2 -no-pie -lm -fopenmp
LDFLAGS = -lm -fopenmp -no-pie

# ---- optimatrix ASM kernels ----------------------------------------
OPT_DIR   = optimatrix
OPT_OBJ   = opt_obj
OPT_SRCS  = $(OPT_DIR)/src/gemv.asm \
            $(OPT_DIR)/src/gemm.asm \
            $(OPT_DIR)/src/gemv_avx2.asm \
            $(OPT_DIR)/src/gemm_avx2.asm \
            $(OPT_DIR)/src/hadamard.asm \
            $(OPT_DIR)/src/activations.asm \
            $(OPT_DIR)/src/scan1d.asm \
            $(OPT_DIR)/src/scan2d.asm
OPT_OBJS  = $(patsubst $(OPT_DIR)/src/%.asm, $(OPT_OBJ)/%.o, $(OPT_SRCS))

$(OPT_OBJ)/%.o: $(OPT_DIR)/src/%.asm
	@mkdir -p $(OPT_OBJ)
	$(NASM) -f elf64 -I $(OPT_DIR)/include/ $< -o $@

# CUDA / cuBLAS flags
# Targets sm_80 (A100/3090), sm_86 (RTX 30xx), sm_89 (RTX 40xx), sm_90 (H100)
NVCCFLAGS = -O3 -std=c++17 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_90,code=sm_90
CUDA_LDFLAGS = -lcublas -lcurand -lm

LARGE_TRAIN_EXECUTABLE = train_large

# Source files
SOURCES = mamba.c main.c
ADVANCED_SOURCES = mamba.c advanced_example.c
MILLION_SOURCES = mamba.c million_params.c
OBJECTS = $(SOURCES:.c=.o)
ADVANCED_OBJECTS = $(ADVANCED_SOURCES:.c=.o)
MILLION_OBJECTS = $(MILLION_SOURCES:.c=.o)
EXECUTABLE = mamba_demo
ADVANCED_EXECUTABLE = mamba_advanced
TRAIN_EXECUTABLE = mamba_train
MILLION_EXECUTABLE = mamba_million
LM_TRAIN_EXECUTABLE = mamba_lm_train
CHAT_EXECUTABLE = mamba_chat

# Default target
all: $(EXECUTABLE) $(ADVANCED_EXECUTABLE) $(MILLION_EXECUTABLE)

# Build training executable
$(TRAIN_EXECUTABLE): mamba.o train.o $(OPT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build LM training executable
$(LM_TRAIN_EXECUTABLE): mamba.o lm.o train_lm.o $(OPT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build chat executable
$(CHAT_EXECUTABLE): mamba.o lm.o chat.o $(OPT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ── CUDA 1B model ───────────────────────────────────────────────────
# Compile CUDA implementation
mamba_cuda.o: mamba_cuda.cu mamba_large.h
	$(NVCC) $(NVCCFLAGS) -c mamba_cuda.cu -o mamba_cuda.o

# Compile C training driver
train_large.o: train_large.c mamba_large.h
	$(CC) -Wall -O2 -std=c99 -c train_large.c -o train_large.o

# Link everything
$(LARGE_TRAIN_EXECUTABLE): mamba_cuda.o train_large.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

# Generate local dataset (no network needed)
dataset:
	python3 generate_data.py

# Download dataset from internet (requires network access)
download-data:
	python3 download_data.py

# Build basic executable
$(EXECUTABLE): mamba.o main.o $(OPT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build advanced executable
$(ADVANCED_EXECUTABLE): mamba.o advanced_example.o $(OPT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build 1M parameter executable
$(MILLION_EXECUTABLE): mamba.o million_params.o $(OPT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(ADVANCED_OBJECTS) $(MILLION_OBJECTS) \
	      $(EXECUTABLE) $(ADVANCED_EXECUTABLE) $(TRAIN_EXECUTABLE) \
	      $(MILLION_EXECUTABLE) $(LM_TRAIN_EXECUTABLE) $(CHAT_EXECUTABLE) \
	      $(LARGE_TRAIN_EXECUTABLE) mamba_cuda.o train_large.o *.o
	rm -rf $(OPT_OBJ)

# Run the basic program
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Run the advanced program
run-advanced: $(ADVANCED_EXECUTABLE)
	./$(ADVANCED_EXECUTABLE)

run-train: $(TRAIN_EXECUTABLE)
	./$(TRAIN_EXECUTABLE)

run-million: $(MILLION_EXECUTABLE)
	./$(MILLION_EXECUTABLE)

run-lm-train: $(LM_TRAIN_EXECUTABLE)
	./$(LM_TRAIN_EXECUTABLE)

run-chat: $(CHAT_EXECUTABLE)
	./$(CHAT_EXECUTABLE)

repl: $(CHAT_EXECUTABLE)
	./$(CHAT_EXECUTABLE)

# Train 1B CUDA model (requires NVIDIA driver + CUDA toolkit)
train-large: $(LARGE_TRAIN_EXECUTABLE)
	./$(LARGE_TRAIN_EXECUTABLE)

# Train small CUDA model for quick iteration (~7M params, fits in 4 GB VRAM)
train-small: $(LARGE_TRAIN_EXECUTABLE)
	./$(LARGE_TRAIN_EXECUTABLE) data/train.txt large_checkpoint_small.bin 10 --small

# Rebuild everything
rebuild: clean all

# Help target
help:
	@echo "Mamba State Space Model in C - Build Commands:"
	@echo "  make                - Build demo, advanced, million targets"
	@echo "  make mamba_train    - Build 1M param SSM trainer"
	@echo "  make mamba_lm_train - Build char-level LM trainer"
	@echo "  make mamba_chat     - Build chat inference binary"
	@echo "  make run            - Build and run basic demo"
	@echo "  make run-train      - Build and run 1M param training"
	@echo "  make run-lm-train   - Build and train the language model"
	@echo "  make run-chat       - Build and launch the interactive REPL"
	@echo "  make repl           - Same as run-chat"
	@echo "── CUDA (requires NVIDIA driver + CUDA toolkit) ──────────────"
	@echo "  make dataset        - Generate local conversation dataset (no network)"
	@echo "  make download-data  - Download DailyDialog from internet"
	@echo "  make train_large    - Build 1B CUDA training binary"
	@echo "  make train-large    - Build + train 1B model (needs ~80 GB VRAM)"
	@echo "  make train-small    - Build + train ~7M model (needs ~4 GB VRAM)"
	@echo "── General ───────────────────────────────────────────────────"
	@echo "  make clean          - Remove all build artifacts"
	@echo "  make rebuild        - Clean and rebuild CPU targets"

.PHONY: all clean run run-advanced run-train run-million run-lm-train \
        run-chat repl rebuild help dataset download-data train-large train-small

