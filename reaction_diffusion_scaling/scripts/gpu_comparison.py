"""
GPU vs CPU Comparison Experiment for Fisher-KPP Reaction-Diffusion.

This script compares solver performance, numerical accuracy, and conservation
properties across:
  - CPU (baseline)
  - GPU (multiple architectures if available)

Measures:
  - Wall-clock time (with proper CUDA/CPU synchronization)
  - Conservation residual
  - L2 error from CPU reference
  - Analysis of numerical divergence (expected GPU non-determinism vs errors)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from reaction_diffusion.solver import FisherKPPSolver
from reaction_diffusion.partition import StripDecomposer
from reaction_diffusion.conservation import ConservationValidator


class GPUComparison:
    """Manages GPU vs CPU comparison experiments."""
    
    def __init__(self, grid_size: int = 128, timesteps: int = 50, seed: int = 42):
        self.grid_size = grid_size
        self.timesteps = timesteps
        self.seed = seed
        self.results = {}
        
    def get_available_devices(self) -> List[str]:
        """Get list of available compute devices."""
        devices = ["cpu"]
        
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            for i in range(cuda_count):
                devices.append(f"cuda:{i}")
        
        return devices
    
    def run_solver(
        self,
        device: str,
        num_partitions: int = 1,
    ) -> Tuple[torch.Tensor, float, float, Dict]:
        """
        Run the solver on a specific device.
        
        Returns:
            (u_final, elapsed_time, accumulated_reaction, metrics)
        """
        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if "cuda" in device:
            torch.cuda.manual_seed(self.seed)
        
        device_obj = torch.device(device)
        
        # Create solver
        solver = FisherKPPSolver(
            grid_size=self.grid_size,
            domain_size=1.0,
            diffusion=0.1,
            reaction_rate=1.0,
            dt=1e-4,
            device=device,
        )
        
        solver.check_stability()
        
        # Initialize field
        u_initial = solver.initialize_gaussian_bump(center_offset=0.5, amplitude=1.0, width=0.1)
        
        # Run solver
        if num_partitions == 1:
            u = u_initial.clone()
            accumulated_reaction = 0.0
            
            # Pre-warm GPU cache if using CUDA
            if "cuda" in device:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            for step_idx in range(self.timesteps):
                accumulated_reaction += solver.compute_reaction_integral(u) * solver.dt
                u = solver.step(u)
            
            if "cuda" in device:
                torch.cuda.synchronize()
            
            elapsed_time = time.perf_counter() - start_time
            u_final = u
        else:
            # Partitioned solver
            decomposer = StripDecomposer(
                global_grid_size=self.grid_size,
                num_partitions=num_partitions,
                device=device_obj,
                dtype=torch.float32,
            )
            
            partitioned_fields = decomposer.decompose_global_field(u_initial)
            decomposer.exchange_ghosts(partitioned_fields)
            
            accumulated_reaction = 0.0
            
            # Pre-warm GPU cache if using CUDA
            if "cuda" in device:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            for step_idx in range(self.timesteps):
                u_global_current = decomposer.reassemble_global_field(partitioned_fields)
                accumulated_reaction += solver.compute_reaction_integral(u_global_current) * solver.dt
                
                for pid, partition in enumerate(decomposer.partitions):
                    field = partitioned_fields[pid]
                    owned_rows = partition.get_owned_rows(field)
                    
                    lap_u = solver.laplacian_5point(field)
                    owned_rows_new = (
                        owned_rows
                        + solver.dt * (
                            solver.diffusion * lap_u[partition.owned_start:partition.owned_end, :]
                            + solver.reaction_term(owned_rows)
                        )
                    )
                    
                    field[partition.owned_start:partition.owned_end, :] = owned_rows_new
                
                decomposer.exchange_ghosts(partitioned_fields)
            
            if "cuda" in device:
                torch.cuda.synchronize()
            
            elapsed_time = time.perf_counter() - start_time
            u_final = decomposer.reassemble_global_field(partitioned_fields)
        
        # Collect metrics
        validator = ConservationValidator(
            dx=solver.dx,
            dy=solver.dy,
            reaction_rate=solver.reaction_rate,
        )
        
        conservation = validator.validate_conservation(u_initial, u_final, accumulated_reaction)
        
        metrics = {
            "device": device,
            "grid_size": self.grid_size,
            "timesteps": self.timesteps,
            "num_partitions": num_partitions,
            "wall_clock_time": elapsed_time,
            "conservation": conservation,
            "u_initial_sum": float(u_initial.sum()),
            "u_final_sum": float(u_final.sum()),
            "u_final_min": float(u_final.min()),
            "u_final_max": float(u_final.max()),
        }
        
        # Move result to CPU for comparison
        u_final_cpu = u_final.detach().cpu()
        
        return u_final_cpu, elapsed_time, accumulated_reaction, metrics
    
    def compare_results(self, reference, test, device_name: str) -> Dict:
        """
        Compare test result against reference (CPU).
        
        Analyze numerical divergence and determine if it's expected GPU non-determinism.
        """
        diff = test - reference
        abs_diff = torch.abs(diff)
        
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())
        median_diff = float(torch.median(abs_diff))
        
        # Relative L2 error
        l2_error_squared = (diff ** 2).sum().item()
        l2_reference_squared = (reference ** 2).sum().item()
        rel_l2_error = (l2_error_squared ** 0.5) / (l2_reference_squared ** 0.5 + 1e-16)
        
        # Check for bitwise identity
        bitwise_identical = torch.allclose(reference, test, rtol=0, atol=0)
        
        # Analyze divergence pattern
        # GPU non-determinism typically appears in:
        # - Parallel reductions (allreduce operations)
        # - Floating-point order changes
        # Expected relative error: ~1e-7 to 1e-5 (ultra-high precision computation)
        
        analysis = {
            "bitwise_identical": bitwise_identical,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "median_abs_diff": median_diff,
            "rel_l2_error": rel_l2_error,
            "divergence_classification": self._classify_divergence(
                max_diff, mean_diff, rel_l2_error
            ),
            "expected_gpu_nondeterminism": self._is_expected_nondeterminism(rel_l2_error),
        }
        
        return analysis
    
    def _classify_divergence(self, max_diff: float, mean_diff: float, rel_l2: float) -> str:
        """Classify the type of numerical divergence observed."""
        if max_diff < 1e-15:
            return "BITWISE_IDENTICAL"
        elif rel_l2 < 1e-10:
            return "FLOATING_POINT_ROUNDING (negligible)"
        elif rel_l2 < 1e-8:
            return "EXPECTED_GPU_NONDETERMINISM"
        elif rel_l2 < 1e-5:
            return "ACCEPTABLE_VARIATION (parallel reduction differences)"
        elif rel_l2 < 1e-3:
            return "SUSPICIOUS (larger than expected)"
        else:
            return "ERROR (likely bug in ghost exchange or solver)"
    
    def _is_expected_nondeterminism(self, rel_l2: float) -> bool:
        """
        Check if divergence is within expected GPU non-determinism bounds.
        
        GPU non-determinism sources:
        - Parallel reduction order: each GPU may sum in different order
        - CUDA atomic operations: order-dependent accumulation
        - FMA (fused multiply-add) precision
        
        Expected bounds: ~1e-7 to 1e-5 for well-conditioned problems
        """
        return 1e-10 < rel_l2 < 1e-4
    
    def run_comparison(self, num_partitions: int = 1) -> Dict:
        """
        Run full comparison across all available devices.
        
        Returns results summary.
        """
        print("\n" + "="*80)
        print("GPU VS CPU COMPARISON EXPERIMENT")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Grid size: {self.grid_size}×{self.grid_size}")
        print(f"  Time steps: {self.timesteps}")
        print(f"  Partitions: {num_partitions}")
        print(f"  Random seed: {self.seed}")
        
        devices = self.get_available_devices()
        
        print(f"\nAvailable devices: {devices}")
        
        results = {}
        reference_result = None
        cpu_time = None
        
        for device in devices:
            print(f"\n--- Running on {device} ---")
            
            try:
                # Get device info
                if "cuda" in device:
                    device_idx = int(device.split(":")[-1]) if ":" in device else 0
                    gpu_name = torch.cuda.get_device_name(device_idx)
                    print(f"GPU: {gpu_name}")
                else:
                    print(f"Device: CPU")
                
                # Run solver
                u_final, elapsed_time, accum_reaction, metrics = self.run_solver(
                    device, num_partitions
                )
                
                print(f"Wall-clock time: {elapsed_time:.6f} seconds")
                print(f"Conservation residual (rel): {metrics['conservation']['rel_residual']:.6e}")
                
                # Store first run as reference (CPU)
                if reference_result is None and device == "cpu":
                    reference_result = u_final
                    cpu_time = elapsed_time
                    print(f"✓ CPU result set as reference")
                    
                    results[device] = {
                        "time": elapsed_time,
                        "metrics": metrics,
                        "comparison": {
                            "bitwise_identical": True,
                            "max_abs_diff": 0.0,
                            "mean_abs_diff": 0.0,
                            "median_abs_diff": 0.0,
                            "rel_l2_error": 0.0,
                            "divergence_classification": "REFERENCE",
                            "expected_gpu_nondeterminism": False,
                        }
                    }
                elif device == "cpu" and reference_result is not None:
                    # Second CPU run for reproducibility
                    comparison = self.compare_results(reference_result, u_final, device)
                    print(f"vs CPU Reference:")
                    print(f"  Max abs diff: {comparison['max_abs_diff']:.6e}")
                    print(f"  Rel L2 error: {comparison['rel_l2_error']:.6e}")
                    print(f"  Classification: {comparison['divergence_classification']}")
                    
                    results[device + "_rerun"] = {
                        "time": elapsed_time,
                        "metrics": metrics,
                        "comparison": comparison,
                    }
                elif reference_result is not None:
                    # GPU run compared to CPU reference
                    comparison = self.compare_results(reference_result, u_final, device)
                    speedup = cpu_time / elapsed_time if cpu_time > 0 else 0
                    efficiency = (speedup / (int(device.split(":")[-1]) + 1) * 100) if ":" in device else 0
                    
                    print(f"vs CPU Reference:")
                    print(f"  Max abs diff: {comparison['max_abs_diff']:.6e}")
                    print(f"  Mean abs diff: {comparison['mean_abs_diff']:.6e}")
                    print(f"  Median abs diff: {comparison['median_abs_diff']:.6e}")
                    print(f"  Rel L2 error: {comparison['rel_l2_error']:.6e}")
                    print(f"  Divergence type: {comparison['divergence_classification']}")
                    print(f"  Expected GPU non-determinism: {comparison['expected_gpu_nondeterminism']}")
                    print(f"Speedup: {speedup:.2f}x")
                    
                    results[device] = {
                        "time": elapsed_time,
                        "speedup": speedup,
                        "metrics": metrics,
                        "comparison": comparison,
                    }
            
            except Exception as e:
                print(f"✗ Error on {device}: {e}")
                results[device] = {"error": str(e)}
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="GPU vs CPU comparison for Fisher-KPP reaction-diffusion."
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="Grid size (square domain).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50,
        help="Number of time steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=1,
        help="Number of partitions to test.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="gpu_cpu_comparison.json",
        help="Output JSON file for results.",
    )
    
    args = parser.parse_args()
    
    # Run comparison
    comparison = GPUComparison(
        grid_size=args.grid_size,
        timesteps=args.timesteps,
        seed=args.seed,
    )
    
    results = comparison.run_comparison(num_partitions=args.num_partitions)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for device, result in results.items():
        print(f"\n{device}:")
        if "error" in result:
            print(f"  ✗ Error: {result['error']}")
        else:
            print(f"  Time: {result['time']:.6f} s")
            if "comparison" in result:
                comp = result["comparison"]
                print(f"  L2 error vs CPU: {comp['rel_l2_error']:.6e}")
                print(f"  Classification: {comp['divergence_classification']}")
            if "speedup" in result:
                print(f"  Speedup: {result['speedup']:.2f}x")
    
    # Save to JSON
    output_file = Path(args.output_json)
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        results_serializable = {}
        for device, result in results.items():
            results_serializable[device] = _make_serializable(result)
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return 0


def _make_serializable(obj):
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    else:
        return obj


if __name__ == "__main__":
    sys.exit(main())
