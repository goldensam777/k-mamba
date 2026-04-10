#!/bin/bash
# Setup script for Kaggle GPU training
# Run this in a Kaggle notebook cell

echo "=== k-mamba Kaggle Setup (500M params) ==="

# 1. Check GPU
nvidia-smi
nvcc --version

# 2. Install Rust (for BPE tokenizer)
echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustc --version

# 3. Install build dependencies
echo "Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq gcc g++ nasm git make

# 3. Clone repo (adjust URL)
cd /kaggle/working
rm -rf k-mamba
git clone https://github.com/goldensam777/k-mamba.git
cd k-mamba

# 4. Build tokenizer BPE Rust
echo "Building Rust BPE tokenizer..."
cd rust_tokenizer
cargo build --release 2>&1 | tail -5
cd ..

# 5. Build k-mamba CUDA
echo "Building k-mamba with CUDA..."
make clean
make lib 2>&1 | tail -10
make models/kmamba_cuda 2>&1 | tail -5

# 5. Verify build
if [ -f models/kmamba_cuda ]; then
    echo "✅ Build successful!"
    ls -lh models/kmamba_cuda
else
    echo "❌ Build failed"
    exit 1
fi

# 6. Prepare data directory
mkdir -p data

echo "=== Setup complete ==="
echo "Usage: ./models/kmamba_cuda train data/your_corpus.txt checkpoint.bin logs"
