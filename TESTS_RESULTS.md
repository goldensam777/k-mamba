UpdateCTestConfiguration  from :/home/samuel-yevi/Dev/optimus/k-mamba/build/DartConfiguration.tcl
UpdateCTestConfiguration  from :/home/samuel-yevi/Dev/optimus/k-mamba/build/DartConfiguration.tcl
Test project /home/samuel-yevi/Dev/optimus/k-mamba/build
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
test 1
      Start  1: optimatrix_kernels_unit

1: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_optimatrix_kernels
1: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
1: Test timeout computed to be: 10000000
1: === optimatrix Kernels Test Suite ===
1: Testing GEMM/GEMV AVX2 implementations
1: 
1: Testing GEMM small matrices (3x4 * 4x2)...
1: PASS: GEMM small matrices
1: Testing GEMM medium matrices (64x128 * 128x256)...
1: PASS: GEMM medium matrices
1: Testing GEMM edge cases...
1: PASS: GEMM edge cases
1: Testing GEMV small (4x3 * 3)...
1: PASS: GEMV small
1: Testing GEMV large (1024x1024 * 1024)...
1: PASS: GEMV large
1: 
1: === Test Results ===
1: Passed: 5/5 tests
1: All tests PASSED!
1: 
1: === Performance Benchmarks ===
1: Benchmarking GEMM performance...
1: Reference: 36.113 sec (0.74 GFLOPS)
1: AVX2:      35.189 sec (0.76 GFLOPS)
1: Speedup:   1.03x
 1/12 Test  #1: optimatrix_kernels_unit ..........   Passed   71.41 sec
test 2
      Start  2: utilitaires_unit

2: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_utilitaires
2: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
2: Test timeout computed to be: 10000000
2: === optimatrix Utilitaires Test Suite ===
2: Testing basic utilitaires available in optimatrix
2: 
2: Testing gradient norm computation...
2: PASS: Gradient norm computation
2: Testing gradient clipping...
2: PASS: Gradient clipping
2: Testing vector addition...
2: PASS: Vector addition
2: Testing vector scaling...
2: PASS: Vector scaling
2: Testing Hadamard product basic...
2: PASS: Hadamard product basic
2: 
2: === Test Results ===
2: Passed: 5/5 tests
2: All tests PASSED!
2: 
2: === Performance Benchmarks ===
2: Benchmarking vector operations...
2: Vector Add: 0.986 sec (1.06 G elems/sec)
 2/12 Test  #2: utilitaires_unit .................   Passed    1.08 sec
test 3
      Start  3: kmamba_inference_working

3: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_kmamba_inference_working
3: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
3: Test timeout computed to be: 10000000
3: === KMamba Inference Test Suite (Working Version) ===
3: Testing basic inference functionality
3: 
3: Testing KMamba config creation...
3: PASS: Config creation
3: Testing token validation...
3: PASS: Token validation
3: Testing logits computation simulation...
3: PASS: Logits computation
3: Testing token selection (argmax)...
3: PASS: Token selection
3: 
3: === Test Results ===
3: Passed: 4/4 tests
3: All tests PASSED!
3: 
3: === Performance Benchmarks ===
3: Benchmarking simple operations...
3: Logits computation: 0.822 sec (0.04 G ops/sec)
 3/12 Test  #3: kmamba_inference_working .........   Passed    0.83 sec
test 4
      Start  4: convnd_integration

4: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_convnd
4: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
4: Test timeout computed to be: 10000000
4: === ConvND Integration Test Suite ===
4: Testing ConvND integration with MambaBlock
4: 
4: Testing basic ConvND operations...
4: PASS: Basic ConvND operations
4: Testing MambaBlock integration with ConvND...
4: PASS: MambaBlock integration with ConvND
4: 
4: === Test Results ===
4: Passed: 2/2 tests
4: All tests PASSED!
4: 
4: === Performance Benchmarks ===
4: Benchmarking ConvND performance...
4: ConvND Performance:
4:   Throughput: 1.44 GB/s
4:   Latency:    0.182 ms/forward
4:   Data size:  1.00 MB
 4/12 Test  #4: convnd_integration ...............   Passed    0.04 sec
test 5
      Start  5: kmamba_inference_e2e

5: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_kmamba_inference
5: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
5: Test timeout computed to be: 10000000
5: === KMamba Inference Test Suite ===
5: Testing complete KMamba inference pipeline
5: 
5: Testing basic KMamba forward pass...
5: PASS: Basic forward pass
5: Testing autoregressive generation...
5: PASS: Autoregressive generation
5: Testing byte-level vocabulary...
5: PASS: Byte-level vocabulary
5: 
5: === Test Results ===
5: Passed: 3/3 tests
5: All tests PASSED!
5: 
5: === Performance Benchmarks ===
5: Benchmarking inference performance...
5: Inference Performance:
5:   Throughput: 3761.33 tokens/sec
5:   Latency:    34.030 ms/forward
5:   Model size: 3343872 parameters
 5/12 Test  #5: kmamba_inference_e2e .............   Passed    3.48 sec
test 6
      Start  6: kmamba_training_e2e

6: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_kmamba_training
6: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
6: Test timeout computed to be: 10000000
6: === KMamba Training Test Suite ===
6: Testing complete KMamba training pipeline
6: 
6: Testing basic KMamba training step...
6: PASS: Basic training step (loss: 5.5357)
6: Testing training convergence...
6: Loss: 5.5455 -> 0.8294 (6.69x reduction)
6: PASS: Training convergence
6: Testing batch training...
6: Batch loss: 5.5492 (4 sequences)
6: PASS: Batch training
6: 
6: === Test Results ===
6: Passed: 3/3 tests
6: All tests PASSED!
6: 
6: === Performance Benchmarks ===
6: Benchmarking training performance...
6: Training Performance:
6:   Throughput: 7829.50 tokens/sec
6:   Latency:    8.174 ms/step
 6/12 Test  #6: kmamba_training_e2e ..............   Passed    0.42 sec
test 7
      Start  7: gradient_utils_unit

7: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_gradient_utils
7: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
7: Test timeout computed to be: 10000000
7: === Gradient Utils Test Suite ===
7: Testing gradient utilities from optimatrix
7: 
7: Testing gradient norm basic cases...
7: PASS: Gradient norm basic cases
7: Testing gradient norm random vectors...
7: PASS: Gradient norm random vectors
7: Testing gradient norm edge cases...
7: PASS: Gradient norm edge cases
7: Testing gradient clipping (no clipping needed)...
7: PASS: Gradient clipping (no clipping)
7: Testing gradient clipping (clipping needed)...
7: PASS: Gradient clipping (with clipping)
7: Testing gradient clipping edge cases...
7: Empty vector clipping handled gracefully
7: PASS: Gradient clipping edge cases
7: Testing gradient clipping (copy version)...
7: PASS: Gradient clipping (copy version)
7: 
7: === Test Results ===
7: Passed: 7/7 tests
7: All tests PASSED!
7: 
7: === Performance Benchmarks ===
7: Benchmarking gradient operations...
7: Gradient Norm: 0.346 sec (0.29 G elems/sec)
7: Gradient Clip: 0.328 sec (0.30 G elems/sec)
 7/12 Test  #7: gradient_utils_unit ..............   Passed    0.74 sec
test 8
      Start  8: optimizers_unit

8: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_optimizers_new
8: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
8: Test timeout computed to be: 10000000
8: === k-mamba Optimizer Test Suite ===
8: Testing MUONCLIP and SGD optimizers
8: 
8: Testing MUONCLIP optimizer basic functionality...
8: PASS: MUONCLIP basic functionality
8: Testing MUONCLIP gradient clipping...
8: PASS: MUONCLIP gradient clipping
8: Testing SGD optimizer basic functionality...
8: PASS: SGD basic functionality
8: Testing optimizer convergence...
8: Loss: 7.017829 -> 0.000000 (3681160960.00x reduction)
8: PASS: Optimizer convergence
8: Testing optimizer weight decay...
8: PASS: Optimizer weight decay
8: 
8: === Test Results ===
8: Passed: 5/5 tests
8: All tests PASSED!
8: 
8: === Performance Benchmarks ===
8: Benchmarking optimizer performance...
8: Optimizer Performance:
8:   Parameters: 10000
8:   Throughput: 58.10 M params/sec
8:   Latency: 0.172 ms/step
 8/12 Test  #8: optimizers_unit ..................   Passed    0.02 sec
test 9
      Start  9: simple_test

9: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_simple
9: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
9: Test timeout computed to be: 10000000
9: === Simple Test Suite ===
9: Testing basic functionality
9: 
9: Testing math functions...
9: sqrt(3^2 + 4^2) = 5.00 (expected: 5.00)
9: PASS: Math functions
9: 
9: Testing memory allocation...
9: Sum of 0..999: 499.50
9: PASS: Memory allocation
9: 
9: === Test Results ===
9: All basic tests PASSED!
9: Build system working correctly!
 9/12 Test  #9: simple_test ......................   Passed    0.00 sec
test 10
      Start 10: OptimizersTest

10: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_optimizers_new
10: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
10: Test timeout computed to be: 10000000
10: === k-mamba Optimizer Test Suite ===
10: Testing MUONCLIP and SGD optimizers
10: 
10: Testing MUONCLIP optimizer basic functionality...
10: PASS: MUONCLIP basic functionality
10: Testing MUONCLIP gradient clipping...
10: PASS: MUONCLIP gradient clipping
10: Testing SGD optimizer basic functionality...
10: PASS: SGD basic functionality
10: Testing optimizer convergence...
10: Loss: 7.017829 -> 0.000000 (3681160960.00x reduction)
10: PASS: Optimizer convergence
10: Testing optimizer weight decay...
10: PASS: Optimizer weight decay
10: 
10: === Test Results ===
10: Passed: 5/5 tests
10: All tests PASSED!
10: 
10: === Performance Benchmarks ===
10: Benchmarking optimizer performance...
10: Optimizer Performance:
10:   Parameters: 10000
10:   Throughput: 67.85 M params/sec
10:   Latency: 0.147 ms/step
10/12 Test #10: OptimizersTest ...................   Passed    0.02 sec
test 11
      Start 11: gpu_simple_test

11: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_gpu_simple
11: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
11: Test timeout computed to be: 10000000
11: === k-mamba GPU Test Suite (Simple) ===
11: Testing basic GPU functionality
11: 
11: Initializing GPU device...
11: GPU: NVIDIA GeForce MX450 (Compute 7.5)
11: Memory: 1.8 GB total, 1.7 GB free
11: Testing basic GPU memory operations...
11: PASS: Basic GPU memory operations
11: Testing GPU vector operations...
11: GPU Vector Add Performance:
11:   Size: 1048576 elements
11:   Bandwidth: 18.88 GB/s
11:   Latency: 0.666 ms
11: PASS: GPU vector operations
11: Testing GPU matrix multiplication...
11: GPU Matrix Multiply Performance:
11:   Size: 512x512x512
11:   GFLOPS: 160.91
11:   Latency: 1.668 ms
11: PASS: GPU matrix multiplication
11: Testing GPU KMamba simulation...
11: GPU KMamba Simulation Performance:
11:   Model: 256 vocab, 512 dim, 128 seq_len
11:   Batch: 4
11:   Throughput: 895113.50 tokens/sec
11:   Latency: 0.572 ms/forward
11: PASS: GPU KMamba simulation
11: 
11: === GPU Test Results ===
11: Passed: 4/4 tests
11: All GPU tests PASSED!
11: 
11: === GPU Device Summary ===
11: Device: NVIDIA GeForce MX450
11: Compute: 7.5
11: Memory: 1.8 GB total, 1.7 GB free
11: Max Threads/Block: 1024
11: Clock Rate: 1.57 GHz
11/12 Test #11: gpu_simple_test ..................   Passed    0.72 sec
test 12
      Start 12: GpuSimpleTest

12: Test command: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests/test_gpu_simple
12: Working Directory: /home/samuel-yevi/Dev/optimus/k-mamba/build/tests
12: Test timeout computed to be: 10000000
12: === k-mamba GPU Test Suite (Simple) ===
12: Testing basic GPU functionality
12: 
12: Initializing GPU device...
12: GPU: NVIDIA GeForce MX450 (Compute 7.5)
12: Memory: 1.8 GB total, 1.7 GB free
12: Testing basic GPU memory operations...
12: PASS: Basic GPU memory operations
12: Testing GPU vector operations...
12: GPU Vector Add Performance:
12:   Size: 1048576 elements
12:   Bandwidth: 18.39 GB/s
12:   Latency: 0.684 ms
12: PASS: GPU vector operations
12: Testing GPU matrix multiplication...
12: GPU Matrix Multiply Performance:
12:   Size: 512x512x512
12:   GFLOPS: 160.36
12:   Latency: 1.674 ms
12: PASS: GPU matrix multiplication
12: Testing GPU KMamba simulation...
12: GPU KMamba Simulation Performance:
12:   Model: 256 vocab, 512 dim, 128 seq_len
12:   Batch: 4
12:   Throughput: 964626.29 tokens/sec
12:   Latency: 0.531 ms/forward
12: PASS: GPU KMamba simulation
12: 
12: === GPU Test Results ===
12: Passed: 4/4 tests
12: All GPU tests PASSED!
12: 
12: === GPU Device Summary ===
12: Device: NVIDIA GeForce MX450
12: Compute: 7.5
12: Memory: 1.8 GB total, 1.7 GB free
12: Max Threads/Block: 1024
12: Clock Rate: 1.57 GHz
12/12 Test #12: GpuSimpleTest ....................   Passed    0.58 sec

100% tests passed, 0 tests failed out of 12

Total Test time (real) =  79.35 sec
