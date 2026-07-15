"""
Grid-size sweep benchmark: 128, 256, 512, 1024, 2048
N=5 runs per grid size on both CPU and GPU with proper CUDA synchronization.
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reaction_diffusion.solver import FisherKPPSolver
from reaction_diffusion.conservation import ConservationValidator


def run_benchmark_single(grid_size, device, timesteps=50, seed=42):
    """
    Run a single benchmark on specified device.
    Returns: (wall_time, conservation_residual, peak_memory_gb)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create solver
    solver = FisherKPPSolver(
        grid_size=grid_size,
        D=0.1,
        k=0.5,
        dt=0.01,
        device=device
    )
    
    # Initialize field
    u = torch.zeros(grid_size, grid_size, dtype=torch.float32, device=device)
    u[grid_size//4:3*grid_size//4, grid_size//4:3*grid_size//4] = 0.5
    
    # For GPU: synchronize before timing
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Time the solver
    start = time.time()
    for _ in range(timesteps):
        u = solver.step(u)
    
    # Synchronize after timing
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    
    wall_time = end - start
    
    # Compute conservation residual
    validator = ConservationValidator()
    mass_init = 0.5 * (grid_size // 2) ** 2  # Initial mass in domain
    mass_final = validator.compute_mass(u)
    expected_change = validator.compute_reaction_integral(u) * timesteps * 0.01
    conservation_residual = abs((mass_final - mass_init) - expected_change)
    
    # Get peak memory
    peak_memory_gb = 0.0
    if device == 'cuda':
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    
    return wall_time, conservation_residual, peak_memory_gb


def run_sweep(grid_sizes, num_runs=5, timesteps=50):
    """
    Run complete grid-size sweep on both CPU and GPU.
    Returns table with statistics.
    """
    results = []
    
    for grid_size in grid_sizes:
        print(f"\n{'='*70}")
        print(f"Grid size: {grid_size}×{grid_size} ({grid_size**2:,} elements)")
        print(f"{'='*70}")
        
        # CPU runs
        print(f"Running {num_runs} CPU benchmarks...")
        cpu_times = []
        cpu_residuals = []
        cpu_memory = 0.0
        
        for run_idx in range(num_runs):
            cpu_time, cpu_residual, _ = run_benchmark_single(
                grid_size, 'cpu', timesteps
            )
            cpu_times.append(cpu_time)
            cpu_residuals.append(cpu_residual)
            print(f"  Run {run_idx+1}: {cpu_time:.4f}s, residual={cpu_residual:.3e}")
        
        cpu_time_median = np.median(cpu_times)
        cpu_residual_median = np.median(cpu_residuals)
        
        # GPU runs
        print(f"Running {num_runs} GPU benchmarks...")
        gpu_times = []
        gpu_residuals = []
        gpu_memories = []
        
        for run_idx in range(num_runs):
            gpu_time, gpu_residual, gpu_memory = run_benchmark_single(
                grid_size, 'cuda', timesteps
            )
            gpu_times.append(gpu_time)
            gpu_residuals.append(gpu_residual)
            gpu_memories.append(gpu_memory)
            print(f"  Run {run_idx+1}: {gpu_time:.4f}s, residual={gpu_residual:.3e}, mem={gpu_memory:.2f} GB")
        
        gpu_time_median = np.median(gpu_times)
        gpu_residual_median = np.median(gpu_residuals)
        gpu_memory_median = np.median(gpu_memories)
        
        # Compute metrics
        speedup = cpu_time_median / gpu_time_median
        
        # GPU vs CPU difference: run one more time to get both results for L2 error
        print(f"Computing GPU vs CPU accuracy...")
        torch.manual_seed(42)
        np.random.seed(42)
        solver_cpu = FisherKPPSolver(grid_size, 0.1, 0.5, 0.01, 'cpu')
        u_cpu = torch.zeros(grid_size, grid_size, dtype=torch.float32, device='cpu')
        u_cpu[grid_size//4:3*grid_size//4, grid_size//4:3*grid_size//4] = 0.5
        
        for _ in range(timesteps):
            u_cpu = solver_cpu.step(u_cpu)
        
        torch.manual_seed(42)
        np.random.seed(42)
        solver_gpu = FisherKPPSolver(grid_size, 0.1, 0.5, 0.01, 'cuda')
        u_gpu = torch.zeros(grid_size, grid_size, dtype=torch.float32, device='cuda')
        u_gpu[grid_size//4:3*grid_size//4, grid_size//4:3*grid_size//4] = 0.5
        
        torch.cuda.synchronize()
        for _ in range(timesteps):
            u_gpu = solver_gpu.step(u_gpu)
        torch.cuda.synchronize()
        
        u_gpu_cpu = u_gpu.cpu()
        l2_error = torch.norm(u_cpu - u_gpu_cpu) / torch.norm(u_cpu)
        l2_error = l2_error.item()
        
        row = {
            'grid_size': grid_size,
            'elements': grid_size ** 2,
            'cpu_time_median_s': cpu_time_median,
            'gpu_time_median_s': gpu_time_median,
            'speedup': speedup,
            'cpu_conservation_residual': cpu_residual_median,
            'gpu_conservation_residual': gpu_residual_median,
            'l2_error_gpu_vs_cpu': l2_error,
            'gpu_memory_gb': gpu_memory_median,
        }
        results.append(row)
        
        print(f"\nSummary for {grid_size}×{grid_size}:")
        print(f"  CPU time (median of {num_runs}):  {cpu_time_median:.4f}s")
        print(f"  GPU time (median of {num_runs}):  {gpu_time_median:.4f}s")
        print(f"  Speedup:                        {speedup:.2f}x")
        print(f"  CPU conservation residual:      {cpu_residual_median:.3e}")
        print(f"  GPU conservation residual:      {gpu_residual_median:.3e}")
        print(f"  L2 error (GPU vs CPU):          {l2_error:.3e}")
        print(f"  GPU peak memory:                {gpu_memory_median:.2f} GB")
    
    return results


def print_table(results):
    """Pretty-print results table."""
    print("\n" + "="*140)
    print("GRID-SIZE SWEEP RESULTS TABLE")
    print("="*140)
    print(f"{'Grid':<12} {'Elements':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} "
          f"{'CPU Res':<14} {'GPU Res':<14} {'L2 Error':<14} {'GPU Mem (GB)':<12}")
    print("-"*140)
    
    for row in results:
        print(f"{row['grid_size']}×{row['grid_size']:<6} {row['elements']:<12} "
              f"{row['cpu_time_median_s']*1000:>10.2f}  {row['gpu_time_median_s']*1000:>10.2f}  "
              f"{row['speedup']:>8.2f}x  {row['cpu_conservation_residual']:>12.3e}  "
              f"{row['gpu_conservation_residual']:>12.3e}  {row['l2_error_gpu_vs_cpu']:>12.3e}  "
              f"{row['gpu_memory_gb']:>10.2f}")
    
    print("="*140)


def save_results(results, output_file):
    """Save results to JSON."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires GPU.")
        sys.exit(1)
    
    print("GPU: ", torch.cuda.get_device_name(0))
    print("PyTorch version:", torch.__version__)
    
    # Run sweep
    grid_sizes = [128, 256, 512, 1024]
    print(f"\nStarting grid-size sweep: {grid_sizes}")
    print(f"N=5 runs per grid size, timesteps=50")
    
    results = run_sweep(grid_sizes, num_runs=5, timesteps=50)
    
    # Print and save
    print_table(results)
    save_results(results, 'grid_sweep_results.json')
    
    # Also save as CSV for easy viewing
    import csv
    with open('grid_sweep_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("Results also saved to: grid_sweep_results.csv")
