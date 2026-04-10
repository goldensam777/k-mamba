#!/usr/bin/env python3
"""
Script d'entraînement k-mamba pour Kaggle
Usage: python train_kaggle.py --data data/corpus.txt --tokens 10000000
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(cmd, description=""):
    """Run shell command and stream output"""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")
    
    print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description='Train k-mamba on Kaggle GPU')
    parser.add_argument('--data', type=str, default='data/corpus.txt',
                        help='Path to training data')
    parser.add_argument('--tokens', type=int, default=10_000_000,
                        help='Total tokens to train on')
    parser.add_argument('--checkpoint', type=str, default='kmamba_kaggle.bin',
                        help='Checkpoint filename')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--model', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Model type to use')
    
    args = parser.parse_args()
    
    # Verify paths
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("Please upload your corpus to Kaggle or specify correct path")
        sys.exit(1)
    
    # Get file size
    data_size = os.path.getsize(args.data)
    print(f"📊 Data file: {args.data}")
    print(f"📏 Size: {data_size:,} bytes ({data_size/1024/1024:.1f} MB)")
    
    # Check if model binary exists
    model_binary = f"models/kmamba_{args.model}"
    if not os.path.exists(model_binary):
        print(f"❌ Model binary not found: {model_binary}")
        print("Run setup first: ./setup_kaggle.sh")
        sys.exit(1)
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Build training command
    cmd = f"./{model_binary} train {args.data} {args.checkpoint} {args.log_dir} --total-tokens {args.tokens}"
    
    print(f"\n🚀 Starting training...")
    print(f"   Model: kmamba_{args.model}")
    print(f"   Tokens: {args.tokens:,}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Logs: {args.log_dir}/")
    print()
    
    start_time = time.time()
    
    # Run training
    returncode = run_command(cmd, "TRAINING")
    
    elapsed = time.time() - start_time
    
    if returncode == 0:
        print(f"\n✅ Training completed in {elapsed/60:.1f} minutes")
        
        # Verify checkpoint
        if os.path.exists(args.checkpoint):
            ckpt_size = os.path.getsize(args.checkpoint)
            print(f"💾 Checkpoint saved: {args.checkpoint} ({ckpt_size/1024/1024:.1f} MB)")
            
            # Copy to output for download
            output_dir = "/kaggle/output"
            if os.path.exists(output_dir):
                import shutil
                shutil.copy(args.checkpoint, output_dir)
                shutil.copytree(args.log_dir, f"{output_dir}/{args.log_dir}", dirs_exist_ok=True)
                print(f"📥 Files copied to {output_dir}/ for download")
        else:
            print("⚠️  Checkpoint not found")
    else:
        print(f"\n❌ Training failed with exit code {returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
