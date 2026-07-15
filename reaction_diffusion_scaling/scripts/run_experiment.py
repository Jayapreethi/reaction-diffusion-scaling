"""
Experiment runner for Fisher-KPP reaction-diffusion scaling study.

This script initializes the simulation, runs partitioned solvers for different
partition counts, validates conservation properties, and compares results.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import torch
import numpy as np

# Add project root to path for package-relative imports
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from reaction_diffusion.solver import FisherKPPSolver
from reaction_diffusion.partition import StripDecomposer
from reaction_diffusion.conservation import ConservationValidator
from reaction_diffusion.visualize import save_heatmap, save_comparison_heatmaps


def run_serial_solver(
    u_initial: torch.Tensor,
    solver: FisherKPPSolver,
    num_steps: int,
    device: str,
) -> Tuple[torch.Tensor, float, float]:
    """
    Run the serial (single-partition) solver and record timing.
    
    Parameters:
        u_initial (torch.Tensor): Initial field.
        solver (FisherKPPSolver): Solver instance.
        num_steps (int): Number of time steps.
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        u_final (torch.Tensor): Final field.
        elapsed_time (float): Wall-clock time in seconds.
        accumulated_reaction_integral (float): Accumulated integral of R(u).
    """
    u = u_initial.clone()
    accumulated_reaction_integral = 0.0
    
    # Synchronize before timing (GPU)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for step_idx in range(num_steps):
        # Accumulate reaction integral before step
        accumulated_reaction_integral += solver.compute_reaction_integral(u) * solver.dt
        
        # Update field
        u = solver.step(u)
    
    # Synchronize after timing (GPU)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time
    
    return u, elapsed_time, accumulated_reaction_integral


def run_partitioned_solver(
    u_initial: torch.Tensor,
    solver: FisherKPPSolver,
    num_partitions: int,
    num_steps: int,
    device: str,
) -> Tuple[torch.Tensor, float, float]:
    """
    Run the partitioned solver with ghost exchange.
    
    Parameters:
        u_initial (torch.Tensor): Initial field.
        solver (FisherKPPSolver): Solver instance.
        num_partitions (int): Number of partitions.
        num_steps (int): Number of time steps.
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        u_final (torch.Tensor): Final reassembled field.
        elapsed_time (float): Wall-clock time in seconds.
        accumulated_reaction_integral (float): Accumulated integral of R(u).
    """
    decomposer = StripDecomposer(
        global_grid_size=solver.grid_size,
        num_partitions=num_partitions,
        device=solver.device,
        dtype=solver.dtype,
    )
    
    # Decompose initial field
    partitioned_fields = decomposer.decompose_global_field(u_initial)
    
    # Exchange initial ghosts
    decomposer.exchange_ghosts(partitioned_fields)
    
    accumulated_reaction_integral = 0.0
    
    # Synchronize before timing (GPU)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for step_idx in range(num_steps):
        # Before updating, accumulate reaction integral (using current state)
        # For now, we'll accumulate a global estimate; ideally we'd accumulate per-partition
        u_global_current = decomposer.reassemble_global_field(partitioned_fields)
        accumulated_reaction_integral += solver.compute_reaction_integral(u_global_current) * solver.dt
        
        # Update each partition independently
        for pid, partition in enumerate(decomposer.partitions):
            field = partitioned_fields[pid]
            owned_rows = partition.get_owned_rows(field)
            
            # Apply solver step to owned rows only
            # We need to construct a temporary field with owned + ghosts for the Laplacian
            # The Laplacian computation will use the ghosts as boundary values
            
            # Compute Laplacian on the entire local field (ghosts will participate)
            lap_u = solver.laplacian_5point(field)
            
            # Compute reaction on owned rows only
            owned_rows_new = (
                owned_rows
                + solver.dt * (
                    solver.diffusion * lap_u[partition.owned_start:partition.owned_end, :]
                    + solver.reaction_term(owned_rows)
                )
            )
            
            # Update owned rows
            field[partition.owned_start:partition.owned_end, :] = owned_rows_new
        
        # Exchange ghosts for next iteration
        decomposer.exchange_ghosts(partitioned_fields)
    
    # Synchronize after timing (GPU)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time
    
    # Reassemble
    u_final = decomposer.reassemble_global_field(partitioned_fields)
    
    return u_final, elapsed_time, accumulated_reaction_integral


def main():
    parser = argparse.ArgumentParser(
        description="Fisher-KPP reaction-diffusion scaling experiment."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device: cpu or cuda.",
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
        default=100,
        help="Number of time steps.",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Partition counts to test (default: 1 2 4).",
    )
    parser.add_argument(
        "--diffusion",
        type=float,
        default=0.1,
        help="Diffusion coefficient D.",
    )
    parser.add_argument(
        "--reaction-rate",
        type=float,
        default=1.0,
        help="Reaction rate k.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step. If None, computed automatically.",
    )
    parser.add_argument(
        "--domain-size",
        type=float,
        default=1.0,
        help="Physical domain size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results and images.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save heatmaps.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results as JSON.",
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up solver
    device = torch.device(args.device)
    
    # Compute time step if not specified
    if args.dt is None:
        # Use a conservative fraction of the stability limit
        dx = args.domain_size / args.grid_size
        stability_limit = (dx ** 2) / (4.0 * args.diffusion)
        args.dt = 0.8 * stability_limit  # Use 80% of stability limit
        print(f"Computed dt = {args.dt:.2e} (stability limit = {stability_limit:.2e})")
    
    solver = FisherKPPSolver(
        grid_size=args.grid_size,
        domain_size=args.domain_size,
        diffusion=args.diffusion,
        reaction_rate=args.reaction_rate,
        dt=args.dt,
        device=args.device,
    )
    
    # Check stability
    try:
        solver.check_stability()
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Initialize field
    u_initial = solver.initialize_gaussian_bump(center_offset=0.5, amplitude=1.0, width=0.1)
    
    # Storage for results
    results = {
        "parameters": {
            "grid_size": args.grid_size,
            "domain_size": args.domain_size,
            "timesteps": args.timesteps,
            "dt": args.dt,
            "diffusion": args.diffusion,
            "reaction_rate": args.reaction_rate,
            "device": args.device,
            "seed": args.seed,
        },
        "experiments": [],
    }
    
    # Conservation validator
    validator = ConservationValidator(
        dx=solver.dx,
        dy=solver.dy,
        reaction_rate=args.reaction_rate,
    )
    
    print("="*70)
    print("Fisher-KPP Reaction-Diffusion Scaling Experiment")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Grid size: {args.grid_size} × {args.grid_size}")
    print(f"Time steps: {args.timesteps}")
    print(f"Diffusion: {args.diffusion}")
    print(f"Reaction rate: {args.reaction_rate}")
    print(f"dt: {args.dt:.2e}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    # Reference solution (single partition)
    print("\n--- Running reference solution (1 partition) ---")
    u_ref, time_ref, accum_reaction_ref = run_serial_solver(
        u_initial, solver, args.timesteps, args.device
    )
    
    print(f"Reference solution time: {time_ref:.4f} s")
    
    # Conservation check for reference
    conservation_ref = validator.validate_conservation(
        u_initial, u_ref, accum_reaction_ref
    )
    
    print("Conservation metrics:")
    print(f"  Initial mass: {conservation_ref['m_initial']:.6f}")
    print(f"  Final mass: {conservation_ref['m_final']:.6f}")
    print(f"  Actual mass change: {conservation_ref['m_change_actual']:.6f}")
    print(f"  Expected mass change: {conservation_ref['m_change_expected']:.6f}")
    print(f"  Absolute residual: {conservation_ref['abs_residual']:.2e}")
    print(f"  Relative residual: {conservation_ref['rel_residual']:.2e}")
    
    # Save reference result metadata
    exp_result_ref = {
        "num_partitions": 1,
        "wall_clock_time": time_ref,
        "speedup": 1.0,
        "efficiency": 1.0,
        "conservation": conservation_ref,
    }
    results["experiments"].append(exp_result_ref)
    
    # Save reference heatmaps if requested
    if args.save_images:
        print("Saving reference heatmaps...")
        vmin = float(u_initial.min())
        vmax = float(u_initial.max())
        
        save_heatmap(
            u_initial,
            str(output_dir / "ref_initial.png"),
            title="Initial State (Reference)",
            vmin=vmin,
            vmax=vmax,
        )
        save_heatmap(
            u_ref,
            str(output_dir / "ref_final.png"),
            title="Final State (Reference)",
            vmin=vmin,
            vmax=vmax,
        )
    
    # Test partitioned solvers
    partitions_to_test = [p for p in args.num_partitions if p != 1]
    
    for num_partitions in partitions_to_test:
        print(f"\n--- Running partitioned solver ({num_partitions} partitions) ---")
        
        u_part, time_part, accum_reaction_part = run_partitioned_solver(
            u_initial, solver, num_partitions, args.timesteps, args.device
        )
        
        print(f"Partitioned solution time: {time_part:.4f} s")
        
        # Conservation check
        conservation_part = validator.validate_conservation(
            u_initial, u_part, accum_reaction_part
        )
        
        print("Conservation metrics:")
        print(f"  Initial mass: {conservation_part['m_initial']:.6f}")
        print(f"  Final mass: {conservation_part['m_final']:.6f}")
        print(f"  Actual mass change: {conservation_part['m_change_actual']:.6f}")
        print(f"  Expected mass change: {conservation_part['m_change_expected']:.6f}")
        print(f"  Absolute residual: {conservation_part['abs_residual']:.2e}")
        print(f"  Relative residual: {conservation_part['rel_residual']:.2e}")
        
        # Comparison with reference
        comparison = validator.compare_fields(u_part, u_ref)
        
        print("Comparison with reference:")
        print(f"  Max abs difference: {comparison['max_abs_diff']:.2e}")
        print(f"  Mean abs difference: {comparison['mean_abs_diff']:.2e}")
        print(f"  Relative L2 error: {comparison['rel_l2_error']:.2e}")
        
        # Speedup and efficiency
        speedup = time_ref / time_part
        efficiency = speedup / num_partitions * 100
        
        print(f"Performance:")
        print(f"  Speedup: {speedup:.4f}x")
        print(f"  Parallel efficiency: {efficiency:.2f}%")
        
        # Store results
        exp_result = {
            "num_partitions": num_partitions,
            "wall_clock_time": time_part,
            "speedup": speedup,
            "efficiency": efficiency,
            "conservation": conservation_part,
            "comparison": comparison,
        }
        results["experiments"].append(exp_result)
        
        # Save comparison heatmaps if requested
        if args.save_images:
            print(f"Saving comparison heatmaps for {num_partitions} partitions...")
            save_comparison_heatmaps(
                u_part,
                u_ref,
                str(output_dir),
                prefix=f"p{num_partitions}",
            )
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Partitions':<12} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12} {'Rel L2 Error':<12}")
    print("-" * 70)
    
    for exp in results["experiments"]:
        num_p = exp["num_partitions"]
        time_s = exp["wall_clock_time"]
        speedup = exp["speedup"]
        efficiency = exp["efficiency"]
        
        if "comparison" in exp:
            rel_l2 = exp["comparison"]["rel_l2_error"]
            rel_l2_str = f"{rel_l2:.2e}"
        else:
            rel_l2_str = "N/A"
        
        print(f"{num_p:<12} {time_s:<12.4f} {speedup:<12.4f} {efficiency:<12.2f}% {rel_l2_str:<12}")
    
    print("="*70)
    
    # Save results as JSON if requested
    if args.save_json:
        json_path = output_dir / "results.json"
        with open(json_path, 'w') as f:
            # Convert tensors to Python floats for JSON serialization
            results_serializable = _make_serializable(results)
            json.dump(results_serializable, f, indent=2)
        print(f"\nResults saved to {json_path}")
    
    return 0


def _make_serializable(obj):
    """Recursively convert non-JSON-serializable objects to JSON-serializable ones."""
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
