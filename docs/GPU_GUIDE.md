# GPU Acceleration Guide

## Overview

This guide explains GPU support, performance characteristics, and best practices for deployment.

## Quick Decision Tree

```
Is your grid size < 256×256 (65K elements)?
├─ YES → Use CPU (GPU overhead not justified)
└─ NO  → Is it < 512×512 (262K elements)?
         ├─ YES → CPU or GPU (similar speed)
         └─ NO  → Use GPU (significant speedup)
```

## GPU Architecture Considerations

### Tesla V100 (Professional)
- **Memory:** 32 GB VRAM
- **Compute:** 7.0 (Volta)
- **Bandwidth:** 900 GB/s
- **Best for:** Large-scale research, production
- **Price:** ~$5K-$8K

### RTX 3090 (High-end Consumer)
- **Memory:** 24 GB VRAM
- **Compute:** 8.6 (Ampere)
- **Bandwidth:** 936 GB/s
- **Best for:** Research labs, enthusiasts
- **Price:** ~$1.5K-$2K

### RTX 4090 (Latest Consumer)
- **Memory:** 24 GB VRAM
- **Compute:** 8.9 (Ada)
- **Bandwidth:** 1152 GB/s
- **Best for:** Cutting-edge research
- **Price:** ~$1.6K-$2K

## Performance Profiling

### Basic Profiling

```python
import time
import torch

solver_gpu = FisherKPPSolver(256, device='cuda')
u_gpu = torch.randn(256, 256, device='cuda')

# Warm up (first run includes overhead)
for _ in range(3):
    u_gpu = solver_gpu.step(u_gpu)

# Measure
torch.cuda.synchronize()  # Critical: wait for GPU
start = time.time()

for _ in range(50):
    u_gpu = solver_gpu.step(u_gpu)

torch.cuda.synchronize()  # Wait for GPU to finish
elapsed = time.time() - start

print(f"GPU time: {elapsed:.3f}s for 50 steps")
print(f"Per-step: {elapsed/50*1000:.2f}ms")
```

### Advanced Profiling with torch.profiler

```python
import torch.profiler

solver_gpu = FisherKPPSolver(256, device='cuda')
u_gpu = torch.randn(256, 256, device='cuda')

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    on_trace_ready=lambda p: print(p.key_averages().table(
        sort_by='cuda_time_total'
    ))
) as prof:
    for _ in range(10):
        u_gpu = solver_gpu.step(u_gpu)
        prof.step()
```

**Output interpretation:**

| Kernel | Count | Avg Time | Total Time | %Total |
|--------|-------|----------|-----------|--------|
| laplacian | 10 | 1.2ms | 12ms | 45% |
| reaction | 10 | 0.8ms | 8ms | 30% |
| synchronize | 10 | 2.5ms | 25ms | 25% |

**Insights:**
- Laplacian dominates (optimize here if needed)
- Synchronization overhead is significant (show GPU speed)
- Compute-bound (GPU good fit)

### Memory Profiling

```python
# Track GPU memory
torch.cuda.reset_peak_memory_stats()

solver_gpu = FisherKPPSolver(512, device='cuda')
u_gpu = torch.randn(512, 512, device='cuda')

for _ in range(100):
    u_gpu = solver_gpu.step(u_gpu)

peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory_gb:.2f} GB")
```

**Memory breakdown (for 512×512 grid):**
- Input field: 512 × 512 × 4 bytes = 1.0 MB
- Temporary fields: ~5-10 MB
- CUDA context: ~100-500 MB
- Total: ~100-600 MB (typically)

## Optimization Strategies

### 1. Batch Multiple Simulations

```python
# Inefficient: one simulation at a time
for trial in range(100):
    u = torch.zeros(128, 128, device='cuda')
    for step in range(50):
        u = solver.step(u)

# Efficient: batch together
batch_size = 8
grids = torch.zeros(batch_size, 128, 128, device='cuda')
for trial in range(100 // batch_size):
    for step in range(50):
        # Process all 8 simultaneously
        grids = torch.stack([
            solver.step(grids[i]) for i in range(batch_size)
        ])
```

**Speedup:** 2-3x due to better GPU utilization

### 2. Fused Kernels

```python
# Current: two separate kernels
lap = solver.laplacian_5point(u)
react = solver.reaction_term(u)
u_new = u + dt * (D * lap + react)

# Fused: single kernel (custom CUDA code)
# u_new = fused_step(u, D, k, dt)
# Saves: memory bandwidth, kernel launch overhead
```

**Potential speedup:** 10-20%

### 3. Mixed Precision

```python
# Use float32 (faster on older GPUs)
u_fp32 = u.float()  # Convert to float32 if using float64

# Or use bfloat16 for newer GPUs
u_bf16 = u.to(torch.bfloat16)
# But verify accuracy still acceptable
```

**Trade-off:** Speed vs. numerical precision

### 4. Asynchronous Operations

```python
# Current: synchronous
u_gpu = solver.step(u_gpu)
u_cpu = u_gpu.cpu()  # Blocks until GPU done

# Asynchronous: don't wait
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    u_gpu = solver.step(u_gpu)
# Do CPU work here while GPU runs
other_cpu_work()
# Then synchronize
torch.cuda.current_stream().wait_stream(stream)
u_cpu = u_gpu.cpu()
```

**Use case:** Multi-GPU pipelines

## Deployment on HPC Clusters

### Talon HPC (University of North Dakota)

#### Submit GPU Job

```bash
sbatch submit_gpu_job.sh
```

#### Check Job Status

```bash
squeue -j JOBID
squeue -u your_username  # All your jobs
```

#### Monitor GPU While Running

```bash
# SSH to compute node directly
ssh talon35  # Get node name from squeue

# Check GPU
nvidia-smi
nvidia-smi -l 1  # Update every 1 second
```

#### Common Issues

**Error:** "Requested node configuration not available"
- Solution: Check available partitions
  ```bash
  sinfo  # Show partitions
  sbatch -p talon-gpu32 submit_gpu_job.sh
  ```

**Error:** "NVIDIA driver not found"
- Solution: Module not loaded, add to batch script:
  ```bash
  module load cuda
  module load nvidia
  ```

**Error:** GPU OOM (Out of Memory)
- Solution: Reduce grid size or enable gradient checkpointing
  ```bash
  sbatch --export=GRID_SIZE=512 submit_gpu_job.sh
  ```

### Multi-GPU on Single Node

```python
# Use all available GPUs
device_ids = list(range(torch.cuda.device_count()))

# For data parallelism
model = torch.nn.DataParallel(solver, device_ids=device_ids)

# Or use NCCL for efficient communication
import torch.distributed as dist
dist.init_process_group('nccl')
```

### Multi-Node (MPI)

For now: Use domain decomposition (partition.py)

Future: Integration with MPI for distributed memory

## Troubleshooting

### GPU Not Detected

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())   # Should be >= 1
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

**Fix:**
```bash
# Check driver
nvidia-smi

# Update PyTorch CUDA version
pip install torch::pytorch::version=2.0 torchvision torchaudio pytorch-cuda=11.8
```

### Numerical Errors on GPU

```python
# Verify physics still holds
validator = ConservationValidator()
residual = validator.validate_conservation(u_init, u_final, steps)
print(f"Conservation residual: {residual:.3e}")  # Should be < 1e-5

# Compare GPU vs CPU
l2_error = validator.compare_fields(u_cpu, u_gpu)
print(f"GPU-CPU L2 error: {l2_error:.3e}")  # Should be < 1e-7
```

### Slow GPU Performance

1. **Check utilization:**
   ```bash
   nvidia-smi -q -d utilization  # Should be >90%
   ```

2. **Check memory:**
   ```bash
   nvidia-smi -q -d memory  # Should be ~80% used
   ```

3. **Profile kernels:**
   ```bash
   nsys profile python scripts/gpu_comparison.py  # NVIDIA Systems tools
   ```

4. **Compare against CPU:**
   ```bash
   python scripts/gpu_comparison.py --device cpu
   ```

## Best Practices

✅ **Do:**
- Benchmark before and after GPU migration
- Use proper CUDA synchronization before timing
- Validate physics (conservation, accuracy)
- Profile to find bottlenecks
- Document performance characteristics
- Test on actual hardware before deployment

❌ **Don't:**
- Assume GPU is always faster
- Skip physics validation
- Use unsynchronized GPU timing
- Optimize without profiling
- Ignore memory constraints
- Deploy without testing

## GPU vs CPU Decision Matrix

| Factor | CPU Better | GPU Better |
|--------|-----------|-----------|
| Grid size | < 256×256 | > 512×512 |
| Problem count | 1-2 | 8+ independent |
| Development | Early stage | Production |
| Precision | High (float64) | Lower (float32) |
| Power budget | Limited | Unlimited |
| Latency | Low | Acceptable |
| Throughput | Low-medium | High |
| Cost per FLOP | High | Low |

## References

- NVIDIA CUDA Programming Guide
- PyTorch CUDA Semantics
- GPU Gems (online articles)

---

**Last Updated:** July 2026
