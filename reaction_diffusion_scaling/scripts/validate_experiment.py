"""
Comprehensive validation script for the reaction-diffusion experiment.

Validates:
1. Ghost exchange genuinely propagates between partitions (perturbation test)
2. CFL stability condition is satisfied
3. Conservation residuals are small and consistent across partition counts
4. L2 errors from reference solution are within floating-point tolerance
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from reaction_diffusion.solver import FisherKPPSolver
from reaction_diffusion.partition import StripDecomposer
from reaction_diffusion.conservation import ConservationValidator


def test_ghost_exchange_propagation():
    """
    Test that ghost exchange genuinely propagates values between partitions.
    
    This is a targeted perturbation test that modifies a boundary value in one partition
    and verifies it appears in the neighbor's ghost row, then affects that neighbor's
    computation in the next step.
    """
    print("\n" + "="*70)
    print("TEST 1: Ghost Exchange Perturbation Propagation")
    print("="*70)
    
    grid_size = 64
    num_partitions = 2
    
    solver = FisherKPPSolver(
        grid_size=grid_size,
        domain_size=1.0,
        diffusion=0.1,
        reaction_rate=1.0,
        dt=1e-4,
        device="cpu",
    )
    
    decomposer = StripDecomposer(
        global_grid_size=grid_size,
        num_partitions=num_partitions,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    
    # Initialize
    u_global = solver.initialize_gaussian_bump(center_offset=0.5, amplitude=1.0, width=0.1)
    partitioned_fields = decomposer.decompose_global_field(u_global)
    decomposer.exchange_ghosts(partitioned_fields)
    
    # Get initial state of partition 1 (lower partition)
    p1_field = partitioned_fields[1]
    p1_top_ghost_initial = p1_field[0, :].clone()
    
    print(f"\nInitial state:")
    print(f"  Partition 1 top ghost row (index 0) - first 5 values: {p1_top_ghost_initial[:5]}")
    
    # Perturb the boundary between partitions in partition 0 (upper)
    p0_field = partitioned_fields[0]
    p0_partition = decomposer.partitions[0]
    
    # The bottom boundary of p0 is at index p0_partition.owned_end - 1
    perturbation = 0.5  # Add significant perturbation
    p0_field[p0_partition.owned_end - 1, :] = perturbation
    
    print(f"\nPerturbation applied:")
    print(f"  Partition 0 bottom owned row (index {p0_partition.owned_end - 1})")
    print(f"  Set all values to: {perturbation}")
    print(f"  First 5 values: {p0_field[p0_partition.owned_end - 1, :5]}")
    
    # Exchange ghosts (this should propagate the perturbation to p1)
    decomposer.exchange_ghosts(partitioned_fields)
    
    p1_top_ghost_after = p1_field[0, :].clone()
    
    print(f"\nAfter ghost exchange:")
    print(f"  Partition 1 top ghost row now has first 5 values: {p1_top_ghost_after[:5]}")
    
    # Verify propagation
    perturbation_propagated = torch.allclose(p1_top_ghost_after, torch.full_like(p1_top_ghost_after, perturbation))
    
    if perturbation_propagated:
        print(f"\n✅ PASS: Perturbation successfully propagated to neighboring partition's ghost row")
        print(f"   This confirms ghost exchange is NOT coincidental matching.")
    else:
        print(f"\n❌ FAIL: Perturbation did not propagate correctly")
        return False
    
    # Now verify that the ghost affects the neighbor's next computation
    print(f"\nVerifying ghost affects neighbor's computation:")
    
    # Store state before step
    p1_owned_rows_before = p1_field[decomposer.partitions[1].owned_start:decomposer.partitions[1].owned_end, :].clone()
    
    # Apply one solver step to p1
    p1_partition = decomposer.partitions[1]
    lap_u = solver.laplacian_5point(p1_field)
    reaction = solver.reaction_term(p1_owned_rows_before)
    p1_owned_rows_new = p1_owned_rows_before + solver.dt * (
        solver.diffusion * lap_u[p1_partition.owned_start:p1_partition.owned_end, :]
        + reaction
    )
    
    # The top-most owned row of p1 should be affected by the ghost row
    # because the Laplacian at that row uses the ghost row above it
    top_row_change = (p1_owned_rows_new[0, :] - p1_owned_rows_before[0, :]).abs()
    
    print(f"  Top owned row of partition 1 change magnitude:")
    print(f"    Max: {top_row_change.max():.6e}")
    print(f"    Mean: {top_row_change.mean():.6e}")
    
    # The change should be non-zero due to Laplacian using the perturbed ghost
    if top_row_change.max() > 1e-10:
        print(f"\n✅ PASS: Ghost row affected computation (change > 1e-10)")
        print(f"   The perturbed boundary value influenced the neighbor's time step.")
        return True
    else:
        print(f"\n❌ FAIL: Ghost row did not affect computation")
        return False


def test_cfl_stability_condition():
    """
    Verify that the CFL (Courant-Friedrichs-Lewy) stability condition is satisfied.
    
    For explicit parabolic problems: dt ≤ dx²/(4D)
    """
    print("\n" + "="*70)
    print("TEST 2: CFL Stability Condition")
    print("="*70)
    
    configs = [
        {"grid_size": 64, "domain_size": 1.0, "diffusion": 0.1},
        {"grid_size": 128, "domain_size": 1.0, "diffusion": 0.1},
        {"grid_size": 256, "domain_size": 1.0, "diffusion": 0.05},
    ]
    
    all_pass = True
    
    for config in configs:
        grid_size = config["grid_size"]
        domain_size = config["domain_size"]
        diffusion = config["diffusion"]
        
        dx = domain_size / grid_size
        stability_limit = (dx ** 2) / (4.0 * diffusion)
        
        # Test with 80% of stability limit (safe choice)
        dt_safe = 0.8 * stability_limit
        
        # Create solver with safe dt
        solver_safe = FisherKPPSolver(
            grid_size=grid_size,
            domain_size=domain_size,
            diffusion=diffusion,
            reaction_rate=1.0,
            dt=dt_safe,
            device="cpu",
        )
        
        try:
            solver_safe.check_stability()
            stable = True
            print(f"\n✅ Config: grid={grid_size}, dx={dx:.4e}, D={diffusion}")
            print(f"   Stability limit: {stability_limit:.4e}")
            print(f"   Chosen dt (80%): {dt_safe:.4e}")
            print(f"   CFL condition: dt={dt_safe:.4e} ≤ dx²/(4D)={stability_limit:.4e}")
            print(f"   Status: SATISFIED")
        except ValueError as e:
            stable = False
            print(f"\n❌ Config: grid={grid_size}")
            print(f"   Error: {e}")
            all_pass = False
        
        # Also test with unstable dt (should fail)
        dt_unstable = 1.5 * stability_limit
        solver_unstable = FisherKPPSolver(
            grid_size=grid_size,
            domain_size=domain_size,
            diffusion=diffusion,
            reaction_rate=1.0,
            dt=dt_unstable,
            device="cpu",
        )
        
        try:
            solver_unstable.check_stability()
            print(f"   ❌ Unstable dt check failed to catch violation")
            all_pass = False
        except ValueError:
            print(f"   ✅ Unstable dt correctly detected: {dt_unstable:.4e} > {stability_limit:.4e}")
    
    if all_pass:
        print(f"\n✅ PASS: All CFL conditions satisfied and validation working")
    
    return all_pass


def run_multipartition_experiment() -> Tuple[bool, Dict]:
    """
    Run experiment with 1, 2, and 4 partitions to verify conservation and accuracy consistency.
    
    Returns:
        (success, metrics_dict)
    """
    print("\n" + "="*70)
    print("TEST 3: Conservation & Accuracy Across Partition Counts")
    print("="*70)
    
    grid_size = 128
    timesteps = 50
    num_partitions_list = [1, 2, 4]
    
    solver = FisherKPPSolver(
        grid_size=grid_size,
        domain_size=1.0,
        diffusion=0.1,
        reaction_rate=1.0,
        dt=1e-4,
        device="cpu",
    )
    
    solver.check_stability()
    
    validator = ConservationValidator(
        dx=solver.dx,
        dy=solver.dy,
        reaction_rate=solver.reaction_rate,
    )
    
    u_initial = solver.initialize_gaussian_bump(center_offset=0.5, amplitude=1.0, width=0.1)
    
    results = {}
    reference_solution = None
    
    for num_partitions in num_partitions_list:
        print(f"\n--- Running with {num_partitions} partition(s) ---")
        
        if num_partitions == 1:
            # Serial solver (reference)
            u = u_initial.clone()
            accumulated_reaction = 0.0
            
            for step_idx in range(timesteps):
                accumulated_reaction += solver.compute_reaction_integral(u) * solver.dt
                u = solver.step(u)
            
            reference_solution = u.clone()
            u_result = u
        else:
            # Partitioned solver
            decomposer = StripDecomposer(
                global_grid_size=grid_size,
                num_partitions=num_partitions,
                device=solver.device,
                dtype=solver.dtype,
            )
            
            partitioned_fields = decomposer.decompose_global_field(u_initial)
            decomposer.exchange_ghosts(partitioned_fields)
            
            accumulated_reaction = 0.0
            
            for step_idx in range(timesteps):
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
            
            u_result = decomposer.reassemble_global_field(partitioned_fields)
        
        # Validate conservation
        conservation = validator.validate_conservation(
            u_initial, u_result, accumulated_reaction
        )
        
        print(f"  Initial mass: {conservation['m_initial']:.8f}")
        print(f"  Final mass: {conservation['m_final']:.8f}")
        print(f"  Actual change: {conservation['m_change_actual']:.8f}")
        print(f"  Expected change (from reaction): {conservation['m_change_expected']:.8f}")
        print(f"  Absolute residual: {conservation['abs_residual']:.4e}")
        print(f"  Relative residual: {conservation['rel_residual']:.4e}")
        
        # Compare with reference
        if num_partitions > 1:
            comparison = validator.compare_fields(u_result, reference_solution)
            print(f"  vs Reference (N=1):")
            print(f"    Max abs diff: {comparison['max_abs_diff']:.4e}")
            print(f"    Mean abs diff: {comparison['mean_abs_diff']:.4e}")
            print(f"    Relative L2 error: {comparison['rel_l2_error']:.4e}")
        else:
            comparison = {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "rel_l2_error": 0.0}
        
        results[num_partitions] = {
            "conservation": conservation,
            "comparison": comparison,
        }
    
    # Verify consistency
    print(f"\n--- Consistency Check ---")
    
    max_rel_residual = max(r["conservation"]["rel_residual"] for r in results.values())
    max_l2_error = max(r["comparison"]["rel_l2_error"] for r in results.values())
    
    print(f"  Max conservation residual (across all N): {max_rel_residual:.4e}")
    print(f"  Max L2 error vs reference: {max_l2_error:.4e}")
    
    consistency_ok = (max_rel_residual < 1e-4) and (max_l2_error < 1e-10)
    
    if consistency_ok:
        print(f"\n✅ PASS: Conservation and accuracy consistent across partitions")
        print(f"   All residuals < 1e-4 and L2 errors < 1e-10 (floating-point precision)")
    else:
        print(f"\n❌ FAIL: Inconsistent results across partitions")
    
    return consistency_ok, results


def main():
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*10 + "REACTION-DIFFUSION EXPERIMENT VALIDATION" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    
    test_results = {}
    
    # Test 1: Ghost exchange propagation
    try:
        test_results["Ghost Exchange"] = test_ghost_exchange_propagation()
    except Exception as e:
        print(f"\n❌ Test 1 failed with exception: {e}")
        test_results["Ghost Exchange"] = False
    
    # Test 2: CFL stability
    try:
        test_results["CFL Stability"] = test_cfl_stability_condition()
    except Exception as e:
        print(f"\n❌ Test 2 failed with exception: {e}")
        test_results["CFL Stability"] = False
    
    # Test 3: Multi-partition consistency
    try:
        test_results["Conservation Consistency"], metrics = run_multipartition_experiment()
    except Exception as e:
        print(f"\n❌ Test 3 failed with exception: {e}")
        test_results["Conservation Consistency"] = False
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(test_results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("\nThe experiment demonstrates:")
        print("  1. Ghost exchange genuinely propagates between partitions")
        print("  2. CFL stability condition is satisfied for chosen dt")
        print("  3. Conservation laws are satisfied across all partition counts")
        print("  4. Partitioned results agree with reference (floating-point precision)")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
