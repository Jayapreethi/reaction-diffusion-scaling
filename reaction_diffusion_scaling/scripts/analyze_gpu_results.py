#!/usr/bin/env python3
"""
Analyze GPU comparison results JSON and generate formatted report.

Usage:
    python analyze_gpu_results.py outputs/cpu_baseline.json outputs/gpu_cuda0.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse


def load_results(json_file: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(json_file) as f:
        return json.load(f)


def format_metric(value: float, decimals: int = 3, scientific: bool = True) -> str:
    """Format a numeric metric for display."""
    if value == 0:
        return "0"
    
    abs_val = abs(value)
    if scientific and (abs_val < 1e-4 or abs_val > 1e6):
        return f"{value:.{decimals}e}"
    else:
        return f"{value:.{decimals}f}"


def analyze_single_result(device_name: str, result: Dict[str, Any]) -> None:
    """Analyze and print a single device result."""
    print(f"\n{'='*80}")
    print(f"Device: {device_name}")
    print(f"{'='*80}")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    metrics = result.get("metrics", {})
    conservation = metrics.get("conservation", {})
    comparison = result.get("comparison", {})
    
    # Device configuration
    print(f"\nConfiguration:")
    print(f"  Grid size: {metrics.get('grid_size', 'N/A')}×{metrics.get('grid_size', 'N/A')}")
    print(f"  Time steps: {metrics.get('timesteps', 'N/A')}")
    print(f"  Partitions: {metrics.get('num_partitions', 'N/A')}")
    
    # Performance
    wall_time = metrics.get("wall_clock_time", 0)
    print(f"\nPerformance:")
    print(f"  Wall-clock time: {wall_time*1e3:.4f} ms")
    print(f"  Grid points: {metrics.get('grid_size', 0)**2:,}")
    print(f"  Iterations/sec: {(metrics.get('timesteps', 0) / wall_time):.1f}")
    
    # Conservation
    print(f"\nConservation Metrics:")
    print(f"  Initial mass: {conservation.get('m_initial', 0):.8e}")
    print(f"  Final mass: {conservation.get('m_final', 0):.8e}")
    print(f"  Actual mass change: {conservation.get('m_change_actual', 0):.8e}")
    print(f"  Expected mass change: {conservation.get('m_change_expected', 0):.8e}")
    print(f"  Absolute residual: {conservation.get('abs_residual', 0):.8e}")
    print(f"  Relative residual: {conservation.get('rel_residual', 0):.8e}")
    
    residual_threshold = 1e-4
    residual_ok = conservation.get('rel_residual', 1) < residual_threshold
    print(f"  ✓ Conservation OK (rel < {residual_threshold})" if residual_ok 
          else f"  ✗ Conservation FAILED (rel ≥ {residual_threshold})")
    
    # Solution statistics
    print(f"\nSolution Statistics:")
    print(f"  u_final min: {metrics.get('u_final_min', 0):.8e}")
    print(f"  u_final max: {metrics.get('u_final_max', 0):.8e}")
    
    # Comparison (if not reference)
    if comparison.get("divergence_classification") != "REFERENCE":
        print(f"\nComparison vs CPU Reference:")
        print(f"  Max absolute difference: {comparison.get('max_abs_diff', 0):.8e}")
        print(f"  Mean absolute difference: {comparison.get('mean_abs_diff', 0):.8e}")
        print(f"  Median absolute difference: {comparison.get('median_abs_diff', 0):.8e}")
        print(f"  Relative L2 error: {comparison.get('rel_l2_error', 0):.8e}")
        print(f"  Divergence classification: {comparison.get('divergence_classification', 'UNKNOWN')}")
        print(f"  Expected GPU non-determinism: {comparison.get('expected_gpu_nondeterminism', False)}")
        
        # Classify divergence
        rel_l2 = comparison.get('rel_l2_error', 1)
        if rel_l2 < 1e-15:
            print(f"  ✓ BITWISE IDENTICAL (no divergence)")
        elif rel_l2 < 1e-10:
            print(f"  ✓ Negligible rounding (acceptable)")
        elif rel_l2 < 1e-5:
            print(f"  ✓ Expected GPU non-determinism (normal)")
        elif rel_l2 < 1e-3:
            print(f"  ⚠ Suspicious variation (investigate)")
        else:
            print(f"  ✗ ERROR level divergence (likely bug)")
        
        speedup = result.get("speedup")
        if speedup:
            print(f"\nPerformance:")
            print(f"  Speedup: {speedup:.1f}x")
            print(f"  Efficiency: {(speedup/1)*100:.1f}% (relative to 1 GPU)")


def compare_devices(cpu_result: Dict, gpu_results: Dict[str, Dict]) -> None:
    """Generate comparison summary across devices."""
    print(f"\n\n{'='*80}")
    print("CROSS-DEVICE COMPARISON")
    print(f"{'='*80}")
    
    devices = {"cpu": cpu_result}
    devices.update(gpu_results)
    
    # Build comparison table
    print(f"\n{'Device':<20} {'Time (ms)':<15} {'Speedup':<12} {'L2 Error':<15} {'Conv Residual':<15}")
    print("-" * 77)
    
    cpu_time = cpu_result.get("metrics", {}).get("wall_clock_time", 1)
    
    for device_name, result in devices.items():
        if "error" in result:
            print(f"{device_name:<20} {'ERROR':<15} {'-':<12} {'-':<15} {'-':<15}")
            continue
        
        time_ms = result.get("metrics", {}).get("wall_clock_time", 0) * 1e3
        speedup = cpu_time / result.get("metrics", {}).get("wall_clock_time", 1)
        
        if device_name == "cpu":
            speedup_str = "1.0x (ref)"
        else:
            speedup_str = f"{speedup:.1f}x"
        
        comparison = result.get("comparison", {})
        l2_error = comparison.get("rel_l2_error", 0)
        l2_str = "N/A" if device_name == "cpu" else f"{l2_error:.2e}"
        
        conservation = result.get("metrics", {}).get("conservation", {})
        residual = conservation.get("rel_residual", 0)
        residual_str = f"{residual:.2e}"
        
        print(f"{device_name:<20} {time_ms:<15.2f} {speedup_str:<12} {l2_str:<15} {residual_str:<15}")
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    all_conservation_ok = True
    for device_name, result in devices.items():
        if "error" not in result:
            residual = result.get("metrics", {}).get("conservation", {}).get("rel_residual", 1)
            if residual >= 1e-4:
                all_conservation_ok = False
                print(f"⚠ {device_name}: Conservation residual too large: {residual:.2e}")
    
    if all_conservation_ok:
        print("✓ All devices: Conservation law satisfied (residual < 1e-4)")
    
    gpu_results_list = list(gpu_results.values())
    if gpu_results_list:
        gpu_l2_errors = [
            r.get("comparison", {}).get("rel_l2_error", 1)
            for r in gpu_results_list
            if "error" not in r
        ]
        if gpu_l2_errors:
            max_l2 = max(gpu_l2_errors)
            if max_l2 < 1e-5:
                print(f"✓ GPU divergence within expected range: max L2 error = {max_l2:.2e}")
            elif max_l2 < 1e-3:
                print(f"⚠ GPU divergence suspicious: max L2 error = {max_l2:.2e} (investigate)")
            else:
                print(f"✗ GPU divergence ERROR level: max L2 error = {max_l2:.2e} (likely bug)")
    
    speedups = []
    for device_name, result in devices.items():
        if device_name != "cpu" and "error" not in result:
            speedup = result.get("speedup", 1)
            speedups.append(speedup)
    
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"✓ Average GPU speedup: {avg_speedup:.1f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GPU comparison results."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="JSON result files to analyze (CPU first, then GPU)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the summary, not individual device details.",
    )
    
    args = parser.parse_args()
    
    if not args.results:
        parser.print_help()
        return 1
    
    # Load all results
    all_results = {}
    for result_file in args.results:
        file_path = Path(result_file)
        if not file_path.exists():
            print(f"Error: File not found: {result_file}")
            return 1
        
        results = load_results(file_path)
        all_results.update(results)
    
    if not all_results:
        print("Error: No results loaded")
        return 1
    
    # Find CPU result
    cpu_result = all_results.get("cpu", {})
    gpu_results = {k: v for k, v in all_results.items() if k != "cpu"}
    
    if not args.summary_only:
        # Print individual results
        if cpu_result:
            analyze_single_result("CPU", cpu_result)
        
        for device_name, result in gpu_results.items():
            analyze_single_result(device_name, result)
    
    # Print summary
    if cpu_result:
        compare_devices(cpu_result, gpu_results)
    else:
        print("Warning: No CPU baseline found for comparison")
    
    print(f"\n{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
