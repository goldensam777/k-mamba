NASM       = nasm
CC         = gcc
NASM_FLAGS = -f elf64 -I include/
CC_FLAGS   = -no-pie -mavx2 -I include
LDFLAGS    = -lm
OBJ_DIR    = obj

ASM_SRCS   = src/gemv.asm        \
             src/gemm.asm        \
             src/gemv_avx2.asm   \
             src/gemm_avx2.asm   \
             src/scan1d.asm      \
             src/scan2d.asm      \
             src/hadamard.asm    \
             src/activations.asm \
             src/scan1d_backward_m1_shared_bc_simple.asm

ASM_OBJS   = $(patsubst src/%.asm, $(OBJ_DIR)/%.o, $(ASM_SRCS))
C_SRCS     = src/scan1d_backward.c src/scan1d_backward_m.c src/convnd.c src/mamba_training.c src/mamba_forward.c
C_OBJS     = $(patsubst src/%.c, $(OBJ_DIR)/%.o, $(C_SRCS))
ALL_OBJS   = $(ASM_OBJS) $(C_OBJS)

.PHONY: all test1 test2 test3 test4 clean

all: $(ALL_OBJS)

test1: $(ALL_OBJS) tests/test_phase1.c
	$(CC) $(CC_FLAGS) tests/test_phase1.c $(ALL_OBJS) -o $(OBJ_DIR)/test1 $(LDFLAGS)

test2: $(ALL_OBJS) tests/test_phase2.c
	$(CC) $(CC_FLAGS) tests/test_phase2.c $(ALL_OBJS) -o $(OBJ_DIR)/test2 $(LDFLAGS)

test3: $(ALL_OBJS) tests/test_phase3.c
	$(CC) $(CC_FLAGS) tests/test_phase3.c $(ALL_OBJS) -o $(OBJ_DIR)/test3 $(LDFLAGS)

test4: $(ALL_OBJS) tests/test_phase4.c
	$(CC) $(CC_FLAGS) tests/test_phase4.c $(ALL_OBJS) -o $(OBJ_DIR)/test4 $(LDFLAGS)

$(OBJ_DIR)/%.o: src/%.asm
	@mkdir -p $(OBJ_DIR)
	$(NASM) $(NASM_FLAGS) $< -o $@

$(OBJ_DIR)/%.o: src/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CC_FLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)
