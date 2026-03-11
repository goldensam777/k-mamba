# BissiMamba

Language model based on the Mamba State Space Model architecture, built entirely on [optimatrix](https://github.com/goldensam777/optimatrix) — a compute engine written in x86-64 assembly and C.

No Python. No GPU. No dependencies beyond libc and libm.

## Architecture

```plain
Input tokens (byte-level, vocab=256)
    |
    v
Embedding lookup [256 x dim]
    |
    v
N x MambaBlock (optimatrix)
    |   scan1d ASM forward
    |   scan1d backward (ASM M=1 / C generic)
    |   MUONCLIP optimizer
    |
    v
LM Head: GEMM AVX2 [dim x 256]
    |
    v
Softmax -> Cross-entropy loss
```

BissiMamba handles model orchestration: embedding, softmax, loss, checkpoint I/O, batch training loop. All heavy compute (GEMM, selective scan, convolution, activations) is delegated to optimatrix's ASM/C kernels.

## Build

Requires: `gcc`, `nasm`, CPU with AVX2 (Intel Haswell+ / AMD Ryzen+).

```bash
make          # builds bissimamba_train + bissimamba_chat
make train    # train only
make chat     # chat only
make clean    # remove all artifacts
```

## Train

```bash
./bissimamba_train <corpus> <checkpoint> <epochs> [steps_per_epoch]

# Example
./bissimamba_train data/conversations.txt checkpoint.bin 200
```

- Byte-level tokenizer (no preprocessing needed — any text file works)
- Batch training (default batch=8) with gradient averaging
- MUONCLIP optimizer on MambaBlocks, SGD+weight decay on embedding/head
- Auto-saves every 10 epochs, resumes from existing checkpoint

### Default config

| Parameter | Value |
|-----------|-------|
| vocab_size | 256 (byte-level) |
| dim | 384 |
| state_size | 1024 |
| seq_len | 128 |
| layers | 1 |
| batch_size | 8 |
| learning rate | 1e-3 |
| optimizer | MUONCLIP (momentum=0.9, beta2=0.999, clip=1.0) |

## Chat

```bash
./bissimamba_chat <checkpoint> [max_tokens] [temperature]

# Example
./bissimamba_chat checkpoint.bin 512 0.8
```

Autoregressive generation, one token at a time. Ctrl-D to quit.

## Project structure

```tree
BissiMamba/
├── bissimamba.h        # Model definition (BissiMamba, BissiMambaConfig)
├── bissimamba.c        # Forward, backward, batch training, checkpoint I/O
├── train_lm.c          # Training CLI
├── chat.c              # Inference CLI
├── Makefile            # Builds optimatrix then BissiMamba
├── data/               # Training corpora
│   ├── conversations.txt
│   └── train.txt
└── optimatrix/         # Compute engine (git submodule)
    ├── src/            # ASM kernels + C orchestration
    ├── include/        # optimatrix.h public API
    └── tests/          # Kernel test suite (phase 1-5)
```

## What comes from where

| BissiMamba (model code) | optimatrix (compute code) |
|------------------------|--------------------------|
| Embedding (memcpy) | GEMM/GEMV AVX2 |
| Softmax, cross-entropy | Selective scan 1D/2D (ASM) |
| Batch training loop | Scan backward (ASM + C) |
| Checkpoint save/load | MambaBlock forward/backward |
| CLI (train, chat) | MUONCLIP optimizer |
| | Activations (SiLU, Sigmoid, ...) |
| | Conv1D/ConvND |
| | Hadamard product |

## License

MIT
