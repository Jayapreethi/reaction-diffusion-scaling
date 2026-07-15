#!/usr/bin/env python3
import torch
import time

print("="*60)
print("PYTORCH CUDA DETECTION TEST")
print("="*60)

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if hasattr(torch.version, 'cuda') else "N/A")
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    print("\n" + "="*60)
    print("GPU PERFORMANCE TEST")
    print("="*60)
    
    torch.cuda.synchronize()
    start = time.time()
    
    x = torch.randn(10000, 10000, device='cuda')
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    
    elapsed = time.time() - start
    print(f"Matrix multiplication (10000x10000): {elapsed*1000:.2f} ms")
    print("✓ GPU is working!")
else:
    print("❌ No GPU detected by PyTorch")
