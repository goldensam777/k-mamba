#!/bin/bash
# Setup script for Google Colab GPU training
# Run this in a Colab notebook cell

echo "=== k-mamba Colab Setup (500M params) ==="

# 1. Check GPU
echo "Checking GPU..."
nvidia-smi
nvcc --version

# 2. Install Rust (for BPE tokenizer)
echo "Installing Rust..."
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi
source $HOME/.cargo/env
rustc --version

# 3. Install build dependencies
echo "Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq gcc g++ nasm git make

# 4. Clone repo
cd /content
rm -rf k-mamba
git clone https://github.com/goldensam777/k-mamba.git
cd k-mamba

# 5. Build tokenizer BPE Rust
echo "Building Rust BPE tokenizer..."
cd rust_tokenizer
cargo build --release 2>&1 | tail -5
cd ..

# 6. Build k-mamba CUDA
echo "Building k-mamba with CUDA..."
make clean
make lib 2>&1 | tail -10
make models/kmamba_cuda 2>&1 | tail -5

# 7. Verify build
if [ -f models/kmamba_cuda ]; then
    echo "✅ Build successful!"
    ls -lh models/kmamba_cuda
else
    echo "❌ Build failed"
    exit 1
fi

# 8. Prepare data directory
mkdir -p data

echo "=== Setup complete ==="
echo "Usage: ./models/kmamba_cuda train data/corpus.txt checkpoint.bin logs"
