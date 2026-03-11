CC         ?= gcc
OPTIMATRIX := optimatrix

CFLAGS  ?= -O3 -no-pie -mavx2 -I $(OPTIMATRIX)/include -Wall -Wextra -Wpedantic
LDFLAGS ?= -lm

OBJ_DIR := obj

BIN_TRAIN := bissimamba_train
BIN_CHAT  := bissimamba_chat

SRCS_TRAIN := train_lm.c bissimamba.c
SRCS_CHAT  := chat.c bissimamba.c

OBJS_TRAIN := $(patsubst %.c,$(OBJ_DIR)/%.o,$(SRCS_TRAIN))
OBJS_CHAT  := $(patsubst %.c,$(OBJ_DIR)/%.o,$(SRCS_CHAT))

.PHONY: all chat train optimatrix clean help

all: train chat

help:
	@echo "  make           Build train + chat"
	@echo "  make train     Build $(BIN_TRAIN)"
	@echo "  make chat      Build $(BIN_CHAT)"
	@echo "  make clean     Remove artifacts"

optimatrix:
	$(MAKE) -C $(OPTIMATRIX) all

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

train: $(BIN_TRAIN)
chat: $(BIN_CHAT)

$(BIN_TRAIN): optimatrix $(OBJS_TRAIN)
	$(CC) $(CFLAGS) $(OBJS_TRAIN) $(OPTIMATRIX)/obj/*.o -o $@ $(LDFLAGS)

$(BIN_CHAT): optimatrix $(OBJS_CHAT)
	$(CC) $(CFLAGS) $(OBJS_CHAT) $(OPTIMATRIX)/obj/*.o -o $@ $(LDFLAGS)

clean:
	$(MAKE) -C $(OPTIMATRIX) clean
	rm -rf $(OBJ_DIR) $(BIN_TRAIN) $(BIN_CHAT)
