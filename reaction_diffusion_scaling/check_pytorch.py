#!/usr/bin/env python
"""Check PyTorch and CUDA availability on Talon."""

import torch

print("=" * 70)
print("TALON PyTorch Environment Check")
print("=" * 70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"PyTorch location: {torch.__file__}")

print("\n--- CUDA Status ---")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"\nGPU Devices:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"\nCUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA Version (runtime): {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("No CUDA devices detected!")

print("\n--- PyTorch Build Info ---")
print(f"OpenMP enabled: {torch.__config__.parallel_info()}")

print("\n--- Available Modules ---")
print(f"torch.nn: {hasattr(torch, 'nn')}")
print(f"torch.optim: {hasattr(torch, 'optim')}")
print(f"torch.cuda: {hasattr(torch, 'cuda')}")
print(f"torch.backends.cudnn: {hasattr(torch.backends, 'cudnn')}")

print("\n" + "=" * 70)
