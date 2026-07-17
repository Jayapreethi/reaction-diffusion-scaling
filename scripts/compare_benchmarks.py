#!/usr/bin/env python3
"""
Quick Benchmark Comparison Tool
Generate summary tables from benchmark results for easy comparison

Usage:
  python scripts/compare_benchmarks.py                    # Compare latest two runs
  python scripts/compare_benchmarks.py --all              # Compare all runs
  python scripts/compare_benchmarks.py --run1 DIR1 --run2 DIR2  # Compare specific runs
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse
import numpy as np


class BenchmarkComparison:
    """Compare benchmark results across multiple runs."""
    
    def __init__(self):
        self.outputs_dir = Path("outputs")
    
    def get_benchmark_runs(self) -> List[Path]:
        """Get all benchmark result directories."""
        if not self.outputs_dir.exists():
            return []
        
        runs = sorted(
            self.outputs_dir.glob("benchmark_*/"),
            key=lambda p: p.name,
            reverse=True  # Most recent first
        )
        return runs
    
    def load_cpu_results(self, run_dir: Path) -> Dict:
        """Load CPU benchmark results from a run."""
        results_file = run_dir / "cpu_results.json"
        if not results_file.exists():
            return None
        
        with open(results_file) as f:
            return json.load(f)
    
    def format_run_name(self, run_path: Path) -> str:
        """Format run directory name for display."""
        return run_path.name.replace("benchmark_", "")
    
    def compare_two_runs(self, run1_path: Path, run2_path: Path) -> None:
        """Compare two benchmark runs."""
        results1 = self.load_cpu_results(run1_path)
        results2 = self.load_cpu_results(run2_path)
        
        if not results1 or not results2:
            print("Error: Could not load results from one or both runs")
            return
        
        name1 = self.format_run_name(run1_path)
        name2 = self.format_run_name(run2_path)
        
        print("\n" + "="*80)
        print(f"Benchmark Comparison: {name1} vs {name2}")
        print("="*80 + "\n")
        
        benchmarks1 = results1.get("benchmarks", [])
        benchmarks2 = results2.get("benchmarks", [])
        
        # Print header
        print(f"{'Grid Size':<15} {'Run 1 (ms)':<15} {'Run 2 (ms)':<15} {'Difference':<15} {'% Change':<10}")
        print("-" * 80)
        
        # Compare each grid size
        for b1, b2 in zip(benchmarks1, benchmarks2):
            grid = f"{b1['grid_size']}×{b1['grid_size']}"
            t1 = b1['median_ms']
            t2 = b2['median_ms']
            diff = t2 - t1
            pct_change = (diff / t1 * 100) if t1 > 0 else 0
            
            status = "↓" if diff < 0 else "↑"
            
            print(
                f"{grid:<15} "
                f"{t1:>13.2f} ms "
                f"{t2:>13.2f} ms "
                f"{status} {abs(diff):>12.2f} ms "
                f"{pct_change:>8.1f}%"
            )
        
        # Overall comparison
        times1 = [b['median_ms'] for b in benchmarks1]
        times2 = [b['median_ms'] for b in benchmarks2]
        avg1 = np.mean(times1)
        avg2 = np.mean(times2)
        
        print("-" * 80)
        print(f"{'AVERAGE':<15} {avg1:>13.2f} ms {avg2:>13.2f} ms {(avg2-avg1):>14.2f} ms {(avg2-avg1)/avg1*100:>8.1f}%")
        print()
    
    def compare_all_runs(self, limit: int = 5) -> None:
        """Compare all recent benchmark runs."""
        runs = self.get_benchmark_runs()[:limit]
        
        if not runs:
            print("No benchmark runs found")
            return
        
        if len(runs) < 2:
            print("Need at least 2 runs to compare")
            return
        
        print("\n" + "="*80)
        print(f"Benchmark History (Last {len(runs)} runs)")
        print("="*80 + "\n")
        
        # Load all results
        all_results = []
        for run in runs:
            results = self.load_cpu_results(run)
            if results:
                all_results.append((self.format_run_name(run), results))
        
        if not all_results:
            print("No CPU results found")
            return
        
        # Get grid sizes from first run
        first_benchmarks = all_results[0][1].get("benchmarks", [])
        
        # Print for each grid size
        for benchmark_idx, first_b in enumerate(first_benchmarks):
            grid_size = first_b['grid_size']
            print(f"\nGrid {grid_size}×{grid_size}:")
            print(f"{'Run':<20} {'Median (ms)':<15} {'Mean (ms)':<15} {'Stdev (ms)':<15}")
            print("-" * 65)
            
            for run_name, results in all_results:
                benchmarks = results.get("benchmarks", [])
                if benchmark_idx < len(benchmarks):
                    b = benchmarks[benchmark_idx]
                    print(
                        f"{run_name:<20} "
                        f"{b['median_ms']:>13.2f} ms "
                        f"{b['mean_ms']:>13.2f} ms "
                        f"{b['stdev_ms']:>13.2f} ms"
                    )
    
    def print_summary(self) -> None:
        """Print summary of available benchmark runs."""
        runs = self.get_benchmark_runs()
        
        if not runs:
            print("No benchmark runs found in outputs/ directory")
            return
        
        print("\nAvailable Benchmark Runs:\n")
        
        for i, run in enumerate(runs[:10]):
            results = self.load_cpu_results(run)
            if results:
                benchmarks = results.get("benchmarks", [])
                timestamp = self.format_run_name(run)
                grid_count = len(benchmarks)
                
                # Get average time
                times = [b['median_ms'] for b in benchmarks]
                avg_time = np.mean(times)
                
                print(f"{i+1}. {timestamp}")
                print(f"   Grid sizes: {grid_count}, Average time: {avg_time:.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CPU benchmark results across runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare latest two runs
  python scripts/compare_benchmarks.py
  
  # List all runs
  python scripts/compare_benchmarks.py --list
  
  # Compare all recent runs
  python scripts/compare_benchmarks.py --all
  
  # Compare specific runs
  python scripts/compare_benchmarks.py --run1 outputs/benchmark_20260716_234006 --run2 outputs/benchmark_20260716_235221
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all available runs")
    parser.add_argument("--all", action="store_true", help="Compare all recent runs")
    parser.add_argument("--run1", help="First run directory")
    parser.add_argument("--run2", help="Second run directory")
    
    args = parser.parse_args()
    
    comparison = BenchmarkComparison()
    
    if args.list:
        comparison.print_summary()
    elif args.run1 and args.run2:
        comparison.compare_two_runs(Path(args.run1), Path(args.run2))
    elif args.all:
        comparison.compare_all_runs()
    else:
        # Compare latest two runs
        runs = comparison.get_benchmark_runs()
        if len(runs) >= 2:
            comparison.compare_two_runs(runs[0], runs[1])
        else:
            comparison.print_summary()


if __name__ == "__main__":
    main()
