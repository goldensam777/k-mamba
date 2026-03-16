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
1: Reference: 38.321 sec (0.70 GFLOPS)
1: AVX2:      34.909 sec (0.77 GFLOPS)
1: Speedup:   1.10x
 1/10 Test  #1: optimatrix_kernels_unit ..........   Passed   73.33 sec
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
2: Vector Add: 0.923 sec (1.14 G elems/sec)
 2/10 Test  #2: utilitaires_unit .................   Passed    1.01 sec
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
3: Logits computation: 0.802 sec (0.04 G ops/sec)
 3/10 Test  #3: kmamba_inference_working .........   Passed    0.81 sec
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
4:   Throughput: 1.53 GB/s
4:   Latency:    0.171 ms/forward
4:   Data size:  1.00 MB
 4/10 Test  #4: convnd_integration ...............   Passed    0.04 sec
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
5:   Throughput: 3998.92 tokens/sec
5:   Latency:    32.009 ms/forward
5:   Model size: 3343872 parameters
 5/10 Test  #5: kmamba_inference_e2e .............   Passed    3.26 sec
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
6:   Throughput: 9468.34 tokens/sec
6:   Latency:    6.759 ms/step
 6/10 Test  #6: kmamba_training_e2e ..............   Passed    0.35 sec
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
7: Gradient Norm: 0.330 sec (0.30 G elems/sec)
7: Gradient Clip: 0.304 sec (0.33 G elems/sec)
 7/10 Test  #7: gradient_utils_unit ..............   Passed    0.69 sec
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
8:   Throughput: 71.51 M params/sec
8:   Latency: 0.140 ms/step
 8/10 Test  #8: optimizers_unit ..................   Passed    0.02 sec
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
 9/10 Test  #9: simple_test ......................   Passed    0.00 sec
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
10:   Throughput: 72.55 M params/sec
10:   Latency: 0.138 ms/step
10/10 Test #10: OptimizersTest ...................   Passed    0.02 sec

100% tests passed, 0 tests failed out of 10

Total Test time (real) =  79.54 sec
