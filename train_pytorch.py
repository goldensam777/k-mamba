#!/usr/bin/env python3
"""
train_pytorch.py — Train BissiMamba in PyTorch, export weights for C inference.

Architecture mirrors lm.c / mamba.c exactly so the exported checkpoint
can be loaded by lm_load() and run by chat.c.

Forward per token x_t ∈ R^dim:
  u_t    = SiLU(W_in @ x_t)                           [state_size]
  delta_t = clamp(softplus(delta_proj @ x_t), dt_min, dt_max)  [scalar]
  A_bar_i = exp(A_log_i)                               [static]
  A_t_i  = exp(delta_t * A_bar_i)
  B_bar_i = (A_t_i - 1) / A_bar_i * B_i  (or delta_t*B_i if A≈0)
  h_t    = A_t * h_{t-1} + B_bar * u_t                [state_size]
  y_t    = W_out @ h_t                                  [dim]
  logit_t = head_W @ y_t + head_bias                    [vocab]
"""

import struct
import sys
import os
import math
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Config  (must match chat.c exactly)
# ─────────────────────────────────────────────────────────────────────────────
VOCAB      = 256
DIM        = 384
STATE_SIZE = 1024
SEQ_LEN    = 128
MAX_GEN    = 256
DT_MIN     = 0.001
DT_MAX     = 0.1

LM_MAGIC   = 0x4C4D3030  # "LM00"

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    def __init__(self, dim, state_size, dt_min=DT_MIN, dt_max=DT_MAX):
        super().__init__()
        self.dim        = dim
        self.state_size = state_size
        self.dt_min     = dt_min
        self.dt_max     = dt_max

        self.W_in       = nn.Parameter(torch.empty(state_size, dim))
        self.W_out      = nn.Parameter(torch.empty(dim, state_size))
        self.A_log      = nn.Parameter(torch.empty(state_size))
        self.B_mat      = nn.Parameter(torch.empty(state_size))
        self.C_mat      = nn.Parameter(torch.empty(state_size))  # kept for compat
        self.delta_proj = nn.Parameter(torch.empty(dim))

        self._init_weights()

    def _init_weights(self):
        # Xavier for W_in, W_out (matches C init)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)
        # A_log: log of negative eigenvalues → A_bar[i] = exp(A_log[i]) < 0?
        # In C: A_log[i] = -exp(spacing * log(dt_scale)) → negative values
        # We init as small negative values
        nn.init.uniform_(self.A_log, -2.0, -0.5)
        # B, C: 1/sqrt(state_size)
        val = 1.0 / math.sqrt(STATE_SIZE)
        nn.init.constant_(self.B_mat, val)
        nn.init.constant_(self.C_mat, val)
        # delta_proj: small uniform
        nn.init.uniform_(self.delta_proj, -0.01, 0.01)

    def forward(self, x):
        """
        x: [T, dim] or [B, T, dim]
        returns y with matching batch shape
        """
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() != 3:
            raise ValueError(f"Expected [T, D] or [B, T, D], got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.state_size)
        ys = []

        # Precompute broadcast-friendly static terms.
        A_bar = self.A_log.exp().unsqueeze(0)             # [1, state_size]
        denom = A_bar.abs().clamp(min=1e-8)
        B_mat = self.B_mat.unsqueeze(0)                   # [1, state_size]

        for t in range(seq_len):
            x_t = x[:, t, :]                              # [B, dim]

            # Controller: u = SiLU(W_in @ x)
            u_t = F.silu(x_t @ self.W_in.t())            # [B, state_size]

            # Delta: softplus(delta_proj · x), clamped
            delta_raw = x_t @ self.delta_proj            # [B]
            delta_t = F.softplus(delta_raw).clamp(
                self.dt_min, self.dt_max
            ).unsqueeze(1)                               # [B, 1]

            # Discretise A and B
            A_t = (delta_t * A_bar).exp()                # [B, state_size]
            B_bar = (A_t - 1.0) / denom * B_mat          # [B, state_size]

            # State update
            h = A_t * h + B_bar * u_t                    # [B, state_size]

            # Output projection
            y_t = h @ self.W_out.t()                     # [B, dim]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                       # [B, T, dim]
        return y.squeeze(0) if squeeze_batch else y


class LM(nn.Module):
    def __init__(self, vocab=VOCAB, dim=DIM, state_size=STATE_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.mamba     = MambaBlock(dim, state_size)
        self.head_W    = nn.Parameter(torch.empty(vocab, dim))
        self.head_bias = nn.Parameter(torch.zeros(vocab))

        nn.init.uniform_(self.embedding.weight, -1/math.sqrt(vocab), 1/math.sqrt(vocab))
        nn.init.xavier_uniform_(self.head_W)

    def forward(self, tokens):
        """
        tokens: [T] int64
        returns logits: [T, vocab]
        """
        x      = self.embedding(tokens)    # [T, dim]
        y      = self.mamba(x)             # [T, dim]
        logits = y @ self.head_W.t() + self.head_bias  # [T, vocab]
        return logits

    def loss(self, tokens):
        """Cross-entropy loss on next-token prediction."""
        if tokens.dim() == 1:
            logits = self.forward(tokens[:-1])           # [T-1, vocab]
            target = tokens[1:]                          # [T-1]
            return F.cross_entropy(logits, target)

        if tokens.dim() != 2:
            raise ValueError(f"Expected [T] or [B, T], got {tuple(tokens.shape)}")

        logits = self.forward(tokens[:, :-1])            # [B, T-1, vocab]
        target = tokens[:, 1:]                           # [B, T-1]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1)
        )

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint export  (binary format expected by lm_load() in lm.c)
# ─────────────────────────────────────────────────────────────────────────────

def export_checkpoint(model, path):
    """Write a binary checkpoint loadable by lm_load()."""
    def w(arr):
        """Write numpy/tensor as float32 bytes."""
        t = arr.detach().cpu().float()
        f.write(t.numpy().tobytes())

    with open(path, 'wb') as f:
        # 1. Magic
        f.write(struct.pack('<I', LM_MAGIC))

        # 2. LMConfig (5 x size_t = 5 x uint64 on 64-bit Linux)
        f.write(struct.pack('<5Q', VOCAB, DIM, STATE_SIZE, SEQ_LEN, MAX_GEN))

        # 3. Embedding table [V x D]
        w(model.embedding.weight)

        # 4. LM head W [V x D]
        w(model.head_W)

        # 5. LM head bias [V]
        w(model.head_bias)

        # 6. A_log [state_size, 1]  → just state_size floats
        w(model.mamba.A_log)

        # 7. B_mat [state_size, 1]
        w(model.mamba.B_mat)

        # 8. C_mat [state_size, 1]
        w(model.mamba.C_mat)

        # 9. W_in [state_size, dim]
        w(model.mamba.W_in)

        # 10. W_out [dim, state_size]
        w(model.mamba.W_out)

        # 11. delta_proj [1, dim]
        w(model.mamba.delta_proj)

    print(f"Checkpoint saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(path):
    with open(path, 'rb') as f:
        raw = f.read()
    return torch.tensor(list(raw), dtype=torch.long)

def make_windows(corpus, seq_len):
    """Slice corpus into overlapping windows of seq_len+1 tokens."""
    step    = seq_len // 2
    windows = []
    for start in range(0, len(corpus) - seq_len, step):
        windows.append(corpus[start : start + seq_len + 1])
    if not windows:
        raise ValueError(f"Corpus too short for seq_len={seq_len}")
    return torch.stack(windows, dim=0)


def resolve_device(device):
    """Map auto/cpu/cuda[:N] to a torch.device with validation."""
    requested = (device or 'auto').lower()
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'

    if requested == 'cpu':
        return torch.device('cpu')

    if requested.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. Check the NVIDIA driver and "
                "install a CUDA-enabled PyTorch build."
            )
        return torch.device(requested)

    raise ValueError(f"Unsupported device '{device}'. Use auto, cpu, or cuda[:N].")

def train(data_path='data/train.txt',
          ckpt_path='lm_checkpoint.bin',
          epochs=20,
          lr=3e-3,
          device=None,
          batch_size=None,
          amp=False):

    device = resolve_device(device)
    if batch_size is None:
        batch_size = 32 if device.type == 'cuda' else 1
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Batch size: {batch_size}")
    print(f"AMP: {'on' if amp and device.type == 'cuda' else 'off'}")

    corpus  = load_corpus(data_path)
    windows = make_windows(corpus, SEQ_LEN)
    print(f"Corpus: {len(corpus)} bytes -> {len(windows)} windows")

    model = LM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=0.01)
    steps_per_epoch = (len(windows) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * steps_per_epoch), eta_min=lr * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == 'cuda')

    best_loss = float('inf')

    for epoch in range(epochs):
        # Shuffle windows each epoch
        idx = torch.randperm(len(windows))
        total_loss, count = 0.0, 0
        epoch_start = time.time()

        for step, start in enumerate(range(0, len(idx), batch_size), start=1):
            batch_idx = idx[start : start + batch_size]
            tokens = windows[batch_idx].to(
                device, non_blocking=(device.type == 'cuda')
            )

            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type='cuda', dtype=torch.float16)
                if amp and device.type == 'cuda' else nullcontext()
            )
            with amp_ctx:
                loss = model.loss(tokens)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            count      += 1

            if step % 25 == 0 or step == steps_per_epoch:
                avg = total_loss / count
                ppl = math.exp(avg)
                print(f"  epoch {epoch:2d}  step {step:5d}/{steps_per_epoch}  "
                      f"loss={avg:.4f}  ppl={ppl:.2f}", flush=True)

        avg_loss = total_loss / count
        ppl      = math.exp(avg_loss)
        elapsed  = time.time() - epoch_start
        print(f"Epoch {epoch:2d}  loss={avg_loss:.4f}  ppl={ppl:.2f}  time={elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            export_checkpoint(model, ckpt_path)
            print(f"  → checkpoint saved (best so far)")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    export_checkpoint(model, ckpt_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',   default='data/train.txt')
    p.add_argument('--ckpt',   default='lm_checkpoint.bin')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr',     type=float, default=3e-3)
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--amp', action='store_true')
    args = p.parse_args()

    train(args.data, args.ckpt, args.epochs, args.lr,
          args.device, args.batch_size, args.amp)
