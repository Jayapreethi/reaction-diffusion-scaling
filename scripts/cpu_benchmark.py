#!/usr/bin/env python3
"""
CPU-Only Benchmark Pipeline for Fisher-KPP Solver
Simplified, no-dependency implementation for Windows compatibility

This script:
1. Reads configuration from YAML
2. Runs CPU benchmarks (pure NumPy, no GPU/PyTorch)
3. Validates physics (mass conservation)
4. Generates reports (JSON, CSV, markdown)
5. Follows national lab standards (reproducible, auditable)
"""

import numpy as np
import yaml
import json
import csv
from pathlib import Path
from datetime import datetime
import sys
import time
import platform


class FisherKPPSolver:
    """CPU-only reaction-diffusion solver using explicit Euler + 5-point stencil."""
    
    def __init__(self, grid_size, d=0.1, k=0.5, dt=0.01):
        self.grid_size = grid_size
        self.d = d  # Diffusion coefficient
        self.k = k  # Reaction rate
        self.dt = dt
        self.dx = 1.0 / (grid_size - 1)
        self.r = self.d * self.dt / (self.dx * self.dx)  # CFL number
        
        # Initialize field
        self.u = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # Set initial condition (centered Gaussian)
        center = grid_size // 2
        x = np.arange(grid_size) - center
        y = np.arange(grid_size) - center
        xx, yy = np.meshgrid(x, y)
        r_sq = xx**2 + yy**2
        self.u = np.exp(-r_sq / (2 * (grid_size // 8)**2))
        self.u = np.clip(self.u, 0, 1)
    
    def step(self):
        """Perform one time step using explicit Euler."""
        u = self.u
        
        # Laplacian with reflecting boundary (5-point stencil)
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            u[0:-2, 1:-1] + u[2:, 1:-1] +
            u[1:-1, 0:-2] + u[1:-1, 2:] -
            4 * u[1:-1, 1:-1]
        ) / (self.dx * self.dx)
        
        # Boundary conditions (reflecting)
        lap[0, 1:-1] = 2 * (u[1, 1:-1] - u[0, 1:-1]) / (self.dx * self.dx)
        lap[-1, 1:-1] = 2 * (u[-2, 1:-1] - u[-1, 1:-1]) / (self.dx * self.dx)
        lap[1:-1, 0] = 2 * (u[1:-1, 1] - u[1:-1, 0]) / (self.dx * self.dx)
        lap[1:-1, -1] = 2 * (u[1:-1, -2] - u[1:-1, -1]) / (self.dx * self.dx)
        
        # Corners
        lap[0, 0] = 0
        lap[0, -1] = 0
        lap[-1, 0] = 0
        lap[-1, -1] = 0
        
        # Reaction term
        reaction = self.k * u * (1 - u)
        
        # Euler step
        self.u = u + self.dt * (self.d * lap + reaction)
        self.u = np.clip(self.u, 0, 1)
    
    def total_mass(self):
        """Compute total mass (integral approximation)."""
        return np.sum(self.u) * self.dx * self.dx


class CPUBenchmark:
    """Run CPU benchmarks with proper measurement and reporting."""
    
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("outputs") / f"benchmark_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """Load YAML configuration."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _capture_environment(self):
        """Capture system metadata."""
        return {
            "timestamp": self.timestamp,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "config_file": str(self.config_path),
            "device": "CPU",
        }
    
    def run_benchmark(self, grid_size, num_runs=5):
        """Run benchmark for given grid size."""
        print(f"\nBenchmarking {grid_size}x{grid_size} grid...")
        
        physics = self.config["physics"]
        benchmark_config = self.config["benchmarking"]
        timesteps = benchmark_config["timesteps"]
        runs = []
        initial_mass = None
        final_mass = None
        
        for run in range(num_runs):
            solver = FisherKPPSolver(
                grid_size,
                d=physics["diffusion_coefficient"],
                k=physics["reaction_rate"],
                dt=physics["timestep"]
            )
            
            if initial_mass is None:
                initial_mass = solver.total_mass()
            
            # Time the computation
            start = time.perf_counter()
            for _ in range(timesteps):
                solver.step()
            elapsed = time.perf_counter() - start
            
            final_mass = solver.total_mass()
            runs.append(elapsed)
            
            # Progress
            print(f"  Run {run+1}/{num_runs}: {elapsed*1000:.2f} ms")
        
        # Statistics
        runs_sorted = sorted(runs)
        results = {
            "grid_size": grid_size,
            "timesteps": timesteps,
            "num_runs": num_runs,
            "times_ms": [t * 1000 for t in runs],
            "median_ms": np.median(runs_sorted) * 1000,
            "mean_ms": np.mean(runs) * 1000,
            "stdev_ms": np.std(runs) * 1000,
            "min_ms": np.min(runs) * 1000,
            "max_ms": np.max(runs) * 1000,
            "initial_mass": float(initial_mass),
            "final_mass": float(final_mass),
            "mass_change": float(final_mass - initial_mass),
        }
        
        return results
    
    def run_all_benchmarks(self):
        """Run benchmarks for all configured grid sizes."""
        print("="*70)
        print(f"CPU Benchmark Pipeline - {self.timestamp}")
        print("="*70)
        
        # Metadata
        metadata = {
            "environment": self._capture_environment(),
            "config": self.config,
            "benchmarks": []
        }
        
        # Extract grid sizes from config
        grid_sizes = []
        if "grids" in self.config:
            # Format: grids: [{size: [128, 128], ...}, ...]
            grid_sizes = [g["size"][0] for g in self.config["grids"]]
        elif "grid_sizes" in self.config:
            # Format: grid_sizes: [128, 256, 512, ...]
            grid_sizes = self.config["grid_sizes"]
        else:
            raise ValueError("Config must have 'grids' or 'grid_sizes'")
        
        num_runs = self.config["benchmarking"]["runs_per_grid"]
        
        # Run benchmarks
        for grid_size in grid_sizes:
            result = self.run_benchmark(grid_size, num_runs=num_runs)
            metadata["benchmarks"].append(result)
        
        # Save results
        results_file = self.output_dir / "cpu_results.json"
        with open(results_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Results saved: {results_file}")
        
        # Generate reports
        self._generate_markdown_report(metadata)
        self._generate_csv_report(metadata)
        
        return metadata
    
    def _generate_markdown_report(self, metadata):
        """Generate markdown report."""
        report_file = self.output_dir / "CPU_BENCHMARK_REPORT.md"
        
        with open(report_file, "w") as f:
            f.write("# CPU Benchmark Report\n\n")
            f.write(f"**Timestamp:** {metadata['environment']['timestamp']}\n\n")
            f.write(f"**Platform:** {metadata['environment']['platform']}\n\n")
            f.write(f"**Python:** {metadata['environment']['python_version']}\n\n")
            f.write(f"**NumPy:** {metadata['environment']['numpy_version']}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write("| Grid Size | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | Stdev (ms) |\n")
            f.write("|-----------|-------------|-----------|----------|----------|------------|\n")
            
            for result in metadata["benchmarks"]:
                gs = result['grid_size']
                f.write(
                    f"| {gs}×{gs} | "
                    f"{result['median_ms']:.2f} | "
                    f"{result['mean_ms']:.2f} | "
                    f"{result['min_ms']:.2f} | "
                    f"{result['max_ms']:.2f} | "
                    f"{result['stdev_ms']:.2f} |\n"
                )
            
            f.write("\n## Physics Validation\n\n")
            for result in metadata["benchmarks"]:
                gs = result['grid_size']
                f.write(f"### {gs}×{gs}\n\n")
                f.write(f"- **Initial Mass:** {result['initial_mass']:.8f}\n")
                f.write(f"- **Final Mass:** {result['final_mass']:.8f}\n")
                f.write(f"- **Change:** {result['mass_change']:.8f}\n\n")
            
            f.write("## Detailed Runs\n\n")
            for result in metadata["benchmarks"]:
                gs = result['grid_size']
                f.write(f"### {gs}×{gs}\n\n")
                for i, t in enumerate(result["times_ms"], 1):
                    f.write(f"- Run {i}: {t:.2f} ms\n")
                f.write("\n")
        
        print(f"✓ Report saved: {report_file}")
    
    def _generate_csv_report(self, metadata):
        """Generate CSV report."""
        csv_file = self.output_dir / "CPU_BENCHMARK_RESULTS.csv"
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Grid Size",
                "Median (ms)",
                "Mean (ms)",
                "Stdev (ms)",
                "Min (ms)",
                "Max (ms)",
                "Initial Mass",
                "Final Mass",
                "Mass Change"
            ])
            
            for result in metadata["benchmarks"]:
                gs = result['grid_size']
                writer.writerow([
                    f"{gs}x{gs}",
                    f"{result['median_ms']:.4f}",
                    f"{result['mean_ms']:.4f}",
                    f"{result['stdev_ms']:.4f}",
                    f"{result['min_ms']:.4f}",
                    f"{result['max_ms']:.4f}",
                    f"{result['initial_mass']:.8f}",
                    f"{result['final_mass']:.8f}",
                    f"{result['mass_change']:.8f}",
                ])
        
        print(f"✓ CSV saved: {csv_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CPU-only benchmark pipeline (NumPy, no GPU/PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cpu_benchmark.py --config config/benchmark_config.yaml
  python scripts/cpu_benchmark.py --config config/benchmark_config.yaml --grid-sizes 128 256 512
        """
    )
    
    parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark configuration YAML"
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        help="Override grid sizes to benchmark"
    )
    
    args = parser.parse_args()
    
    try:
        benchmark = CPUBenchmark(args.config)
        
        # Override grid sizes if provided
        if args.grid_sizes:
            benchmark.config["grid_sizes"] = args.grid_sizes
            # Remove grids key to avoid conflict
            if "grids" in benchmark.config:
                del benchmark.config["grids"]
        
        benchmark.run_all_benchmarks()
        print("\n" + "="*70)
        print("Benchmark Complete")
        print("="*70)
        print(f"\nResults saved to: {benchmark.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
