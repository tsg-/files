#!/usr/bin/env python3
"""
Short PyTorch test script for ROCm
Tests basic GPU availability and operations
"""

import torch
import sys

def test_rocm():
    print("=" * 60)
    print("PyTorch ROCm Test")
    print("=" * 60)

    # Check PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")

    # Check if ROCm is available
    print(f"ROCm available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nERROR: ROCm/CUDA not available!")
        print("Make sure PyTorch is built with ROCm support")
        sys.exit(1)

    # Get device info
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")

    for i in range(device_count):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")

    # Test basic operations
    print("\n" + "=" * 60)
    print("Running basic GPU operations...")
    print("=" * 60)

    device = torch.device("cuda:0")

    # Test 1: Tensor creation on GPU
    print("\n1. Creating tensors on GPU...")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    print(f"   Tensor a shape: {a.shape}, device: {a.device}")
    print(f"   Tensor b shape: {b.shape}, device: {b.device}")

    # Test 2: Matrix multiplication
    print("\n2. Testing matrix multiplication...")
    c = torch.matmul(a, b)
    print(f"   Result shape: {c.shape}")
    print(f"   Sample value: {c[0, 0].item():.4f}")

    # Test 3: Memory allocation and deallocation
    print("\n3. Testing memory management...")
    initial_memory = torch.cuda.memory_allocated(device)
    print(f"   Initial memory allocated: {initial_memory / 1024**2:.2f} MB")

    large_tensor = torch.randn(5000, 5000, device=device)
    after_alloc = torch.cuda.memory_allocated(device)
    print(f"   After allocation: {after_alloc / 1024**2:.2f} MB")

    del large_tensor
    torch.cuda.empty_cache()
    after_free = torch.cuda.memory_allocated(device)
    print(f"   After cleanup: {after_free / 1024**2:.2f} MB")

    # Test 4: Simple neural network operation
    print("\n4. Testing neural network layer...")
    linear = torch.nn.Linear(1000, 500).to(device)
    x = torch.randn(32, 1000, device=device)
    output = linear(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

    # Test 5: GPU synchronization
    print("\n5. Testing GPU synchronization...")
    torch.cuda.synchronize()
    print("   Synchronization successful")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_rocm()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
