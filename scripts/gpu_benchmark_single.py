#!/usr/bin/env python3
"""
GPU Benchmark Single Grid Script for Talon HPC
Executed by SLURM job, produces JSON output for aggregation

Usage:
  python3 gpu_benchmark_single.py --grid-size 512 512 --timesteps 50 --runs 5 --output results.json
"""

import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

try:
    import torch
    import numpy as np
except ImportError:
    print("Error: PyTorch required")
    sys.exit(1)


def run_gpu_benchmark(grid_size: tuple, timesteps: int, runs: int, seed: int = 42, validate: bool = True) -> dict:
    """Run GPU benchmark with reproducibility and validation."""
    
    print(f"GPU Benchmark: {grid_size[0]}×{grid_size[1]} grid, {timesteps} timesteps, {runs} runs")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Import solver
    try:
        from reaction_diffusion.solver import FisherKPPSolver
        from reaction_diffusion.conservation import ConservationValidator
    except ImportError:
        print("Error: reaction_diffusion modules not found")
        sys.exit(1)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Initialize solver
    solver = FisherKPPSolver(
        grid_size=grid_size[0],
        D=0.1,
        k=0.5,
        dt=0.01,
        device='cuda'
    )
    
    # Initialize field
    u = torch.randn(*grid_size).abs() * 0.5
    u = u.to('cuda')
    
    # Run benchmark with proper CUDA synchronization
    run_times = []
    final_fields = []
    
    for run in range(runs):
        # Reset field
        np.random.seed(seed + run)
        u = torch.randn(*grid_size).abs() * 0.5
        u = u.to('cuda')
        
        # Synchronize before timing
        torch.cuda.synchronize()
        start = time.time()
        
        # Run simulation
        for _ in range(timesteps):
            u = solver.step(u)
        
        # Synchronize after timing
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        run_times.append(elapsed)
        final_fields.append(u.cpu().numpy().copy())
        
        print(f"  Run {run+1}/{runs}: {elapsed:.2f} ms")
    
    # Compute statistics
    run_times = np.array(run_times)
    stats = {
        "grid_size": grid_size,
        "elements": grid_size[0] * grid_size[1],
        "device": "cuda",
        "timesteps": timesteps,
        "runs": runs,
        "run_times_ms": run_times.tolist(),
        "median_time_ms": float(np.median(run_times)),
        "mean_time_ms": float(np.mean(run_times)),
        "std_time_ms": float(np.std(run_times)),
        "min_time_ms": float(np.min(run_times)),
        "max_time_ms": float(np.max(run_times)),
    }
    
    # Physics validation if requested
    if validate:
        print("\nPhysics Validation:")
        
        # Re-run single simulation for field analysis
        np.random.seed(seed)
        u_initial = torch.randn(*grid_size).abs() * 0.5
        u_initial = u_initial.to('cuda')
        
        validator = ConservationValidator()
        u_final = u_initial.clone()
        for _ in range(timesteps):
            u_final = solver.step(u_final)
        
        # Check conservation
        initial_mass = solver.compute_mass(u_initial).item()
        final_mass = solver.compute_mass(u_final).item()
        
        conservation_check = validator.check_mass_conservation(
            u_initial.cpu(),
            u_final.cpu(),
            solver.D,
            solver.k,
            solver.dt
        )
        
        stats["physics"] = {
            "initial_mass": float(initial_mass),
            "final_mass": float(final_mass),
            "mass_change": float(final_mass - initial_mass),
            "conservation_residual": float(conservation_check.get("residual", 0)),
            "conservation_valid": bool(conservation_check.get("valid", False))
        }
        
        print(f"  Initial mass: {initial_mass:.8f}")
        print(f"  Final mass: {final_mass:.8f}")
        print(f"  Conservation residual: {stats['physics']['conservation_residual']:.2e}")
    
    print(f"\nSummary:")
    print(f"  Median: {stats['median_time_ms']:.2f} ms")
    print(f"  Mean ± Std: {stats['mean_time_ms']:.2f} ± {stats['std_time_ms']:.2f} ms")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="GPU Benchmark for Single Grid Size"
    )
    parser.add_argument("--grid-size", type=int, nargs=2, required=True,
                        help="Grid dimensions (e.g., 512 512)")
    parser.add_argument("--timesteps", type=int, default=50,
                        help="Number of simulation timesteps")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of benchmark runs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="gpu_results.json",
                        help="Output JSON file")
    parser.add_argument("--validate", action="store_true",
                        help="Perform physics validation")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_gpu_benchmark(
        grid_size=tuple(args.grid_size),
        timesteps=args.timesteps,
        runs=args.runs,
        seed=args.seed,
        validate=args.validate
    )
    
    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_file}")
    
    # Return success exit code
    sys.exit(0)


if __name__ == "__main__":
    main()
