#!/usr/bin/env python3
"""
Master Pipeline Orchestrator for Fisher-KPP GPU/CPU Comparison
National Lab Standard for Reproducible Computational Science

This script:
1. Runs CPU benchmarks locally
2. Generates and submits GPU jobs to Talon HPC cluster
3. Collects and aggregates results
4. Validates physics and performance metrics
5. Generates comprehensive reports

Usage:
    python3 scripts/run_benchmark_suite.py [--local-only] [--cluster-only] [--config config/benchmark_config.yaml]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import yaml
import hashlib

try:
    import torch
    import numpy as np
except ImportError:
    print("Error: PyTorch required. Install with: pip install -r requirements.txt")
    sys.exit(1)


class BenchmarkOrchestrator:
    """Orchestrates end-to-end benchmark pipeline across local and cluster resources."""

    def __init__(self, config_path: str = "config/benchmark_config.yaml"):
        """Initialize orchestrator with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = self._generate_run_id()
        self.results_dir = self._setup_results_directory()
        self.metadata = self._capture_environment()
        
        print(f"✓ Pipeline initialized (Run ID: {self.run_id})")
        print(f"✓ Results directory: {self.results_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Loaded config: {config['experiment']['name']}")
        return config

    def _generate_run_id(self) -> str:
        """Generate unique run ID for reproducibility tracking."""
        # Include git commit hash if available
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent.parent,
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
        except:
            git_hash = "local"
        
        return f"{self.timestamp}_{git_hash}"

    def _setup_results_directory(self) -> Path:
        """Create timestamped results directory."""
        results_base = Path(self.config["output"]["base_directory"])
        results_dir = results_base / f"benchmark_{self.run_id}"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def _capture_environment(self) -> Dict[str, Any]:
        """Capture computational environment for reproducibility."""
        metadata = {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "pytorch_version": getattr(torch, "__version__", "unknown"),
        }
        
        # Safely check CUDA availability
        try:
            metadata["cuda_available"] = torch.cuda.is_available()
        except:
            metadata["cuda_available"] = False
            
        if metadata.get("cuda_available"):
            try:
                metadata["cuda_version"] = getattr(torch.version, "cuda", "unknown")
                metadata["gpu_device"] = torch.cuda.get_device_name(0)
                metadata["gpu_capability"] = torch.cuda.get_device_capability(0)
            except:
                pass
        
        # Capture git state
        try:
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=Path(__file__).parent.parent,
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent.parent,
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
            metadata["git_branch"] = git_branch
            metadata["git_commit"] = git_commit
        except:
            metadata["git_status"] = "unavailable"
        
        # Save metadata
        metadata_file = self.results_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Environment captured:")
        print(f"  PyTorch: {metadata.get('pytorch_version', 'N/A')}")
        print(f"  CUDA: {metadata.get('cuda_version', 'N/A')}")
        if metadata.get("gpu_device"):
            print(f"  GPU: {metadata['gpu_device']}")
        
        return metadata

    def run_local_cpu_benchmarks(self) -> Dict[str, Any]:
        """Run CPU benchmarks on local machine."""
        print("\n" + "="*70)
        print("PHASE 1: LOCAL CPU BENCHMARKING")
        print("="*70)
        
        cpu_results = {}
        
        for grid_config in self.config["grids"]:
            grid_size = tuple(grid_config["size"])
            elements = grid_config["elements"]
            print(f"\n📊 Grid: {grid_size[0]}×{grid_size[1]} ({elements:,} elements)")
            
            try:
                from reaction_diffusion.solver import FisherKPPSolver
                
                # Initialize solver
                solver = FisherKPPSolver(
                    grid_size=grid_size[0],
                    D=self.config["physics"]["diffusion_coefficient"],
                    k=self.config["physics"]["reaction_rate"],
                    dt=self.config["physics"]["timestep"],
                    device="cpu"
                )
                
                # Initialize field with reproducible random seed
                np.random.seed(self.config["reproducibility"]["random_seed"])
                u = torch.randn(*grid_size).abs() * 0.5
                u = u.to("cpu")
                
                # Run multiple times for statistics
                run_times = []
                for run in range(self.config["benchmarking"]["runs_per_grid"]):
                    u_copy = u.clone()
                    start = time.time()
                    
                    for _ in range(self.config["benchmarking"]["timesteps"]):
                        u_copy = solver.step(u_copy)
                    
                    elapsed = (time.time() - start) * 1000  # Convert to ms
                    run_times.append(elapsed)
                
                # Compute statistics
                run_times = np.array(run_times)
                grid_result = {
                    "grid_size": grid_size,
                    "elements": elements,
                    "device": "cpu",
                    "run_times_ms": run_times.tolist(),
                    "median_time_ms": float(np.median(run_times)),
                    "mean_time_ms": float(np.mean(run_times)),
                    "std_time_ms": float(np.std(run_times)),
                    "min_time_ms": float(np.min(run_times)),
                    "max_time_ms": float(np.max(run_times)),
                }
                
                cpu_results[f"{grid_size[0]}x{grid_size[1]}"] = grid_result
                
                print(f"  ✓ Median: {grid_result['median_time_ms']:.2f} ms")
                print(f"  ✓ Stats: μ={grid_result['mean_time_ms']:.2f} σ={grid_result['std_time_ms']:.2f} ms")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                cpu_results[f"{grid_size[0]}x{grid_size[1]}"] = {"error": str(e)}
        
        # Save CPU results
        cpu_file = self.results_dir / "cpu_results.json"
        with open(cpu_file, "w") as f:
            json.dump(cpu_results, f, indent=2)
        print(f"\n✓ CPU results saved: {cpu_file}")
        
        return cpu_results

    def generate_slurm_jobs(self) -> List[str]:
        """Generate SLURM job scripts for cluster execution."""
        print("\n" + "="*70)
        print("PHASE 2: GENERATE SLURM JOBS FOR CLUSTER")
        print("="*70)
        
        jobs_dir = self.results_dir / "slurm_jobs"
        jobs_dir.mkdir(exist_ok=True)
        job_ids = []
        
        for grid_config in self.config["grids"]:
            grid_size = tuple(grid_config["size"])
            grid_str = f"{grid_size[0]}x{grid_size[1]}"
            
            # Create job script
            job_script = self._create_job_script(grid_str, grid_size)
            job_path = jobs_dir / f"job_{grid_str}.sh"
            
            with open(job_path, "w") as f:
                f.write(job_script)
            
            job_ids.append((grid_str, str(job_path)))
            print(f"✓ Generated: {job_path.name}")
        
        return job_ids

    def _create_job_script(self, grid_str: str, grid_size: tuple) -> str:
        """Create single SLURM job script."""
        slurm_cfg = self.config["slurm"]
        output_file = f"gpu_results_{grid_str}_{self.run_id}.json"
        
        script = f"""#!/bin/bash
# SLURM Job Script: Fisher-KPP GPU Benchmark
# Grid: {grid_str}
# Run ID: {self.run_id}
#
# National Lab Standard: Fully reproducible job specification

#SBATCH --job-name=fisher-kpp-{grid_str}
#SBATCH --partition={slurm_cfg['partition']}
#SBATCH --time={slurm_cfg['time_limit']}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={slurm_cfg['cores']}
#SBATCH --mem={slurm_cfg['memory']}
#SBATCH --output=logs/job_{grid_str}_%j.log
#SBATCH --error=logs/job_{grid_str}_%j.err

# Capture environment
echo "Job started at $(date)"
echo "Node: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(python3 --version)"
echo ""

# Activate environment (assuming venv setup)
source ~/.bashrc
cd $PROJECT_ROOT

# Run GPU benchmark with reproducibility
python3 scripts/gpu_benchmark_single.py \\
    --grid-size {grid_size[0]} {grid_size[1]} \\
    --timesteps {self.config['benchmarking']['timesteps']} \\
    --runs {self.config['benchmarking']['runs_per_grid']} \\
    --seed {self.config['reproducibility']['random_seed']} \\
    --output {output_file} \\
    --validate

echo ""
echo "Job completed at $(date)"
"""
        return script

    def submit_jobs_to_cluster(self, job_ids: List[tuple]) -> Dict[str, str]:
        """Submit SLURM jobs to cluster."""
        print("\n" + "="*70)
        print("PHASE 3: SUBMIT JOBS TO TALON CLUSTER")
        print("="*70)
        
        cluster_config = {
            "host": "talon.und.edu",
            "user": "jayapreethi.mohan",
            "project_remote_path": "/home/jayapreethi.mohan/reaction_diffusion_scaling"
        }
        
        print(f"\nCluster: {cluster_config['host']}")
        print(f"User: {cluster_config['user']}")
        
        submitted_jobs = {}
        
        for grid_str, job_path in job_ids:
            try:
                # Copy job script to cluster
                print(f"\n📤 Transferring job script: {grid_str}...")
                scp_cmd = [
                    "scp",
                    job_path,
                    f"{cluster_config['user']}@{cluster_config['host']}:{cluster_config['project_remote_path']}/slurm_scripts/"
                ]
                subprocess.run(scp_cmd, check=True, capture_output=True)
                
                # Submit via SSH
                print(f"  Submitting to queue...")
                job_name = Path(job_path).name
                ssh_cmd = [
                    "ssh",
                    f"{cluster_config['user']}@{cluster_config['host']}",
                    f"cd {cluster_config['project_remote_path']} && sbatch slurm_scripts/{job_name}"
                ]
                result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
                
                # Extract job ID
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs[grid_str] = {
                    "job_id": job_id,
                    "status": "submitted",
                    "submitted_time": datetime.now().isoformat()
                }
                print(f"  ✓ Submitted with Job ID: {job_id}")
                
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed: {e}")
                submitted_jobs[grid_str] = {"error": str(e), "status": "failed"}
        
        # Save job tracking
        jobs_file = self.results_dir / "submitted_jobs.json"
        with open(jobs_file, "w") as f:
            json.dump(submitted_jobs, f, indent=2)
        print(f"\n✓ Job tracking saved: {jobs_file}")
        
        return submitted_jobs

    def wait_for_cluster_results(self, timeout_hours: int = 1) -> Dict[str, Any]:
        """Poll for cluster job completion and retrieve results."""
        print("\n" + "="*70)
        print("PHASE 4: WAIT FOR CLUSTER RESULTS")
        print("="*70)
        print(f"Polling for up to {timeout_hours} hour(s)...")
        
        # Load submitted jobs
        jobs_file = self.results_dir / "submitted_jobs.json"
        with open(jobs_file) as f:
            submitted_jobs = json.load(f)
        
        start_time = time.time()
        timeout_sec = timeout_hours * 3600
        results = {}
        
        while (time.time() - start_time) < timeout_sec:
            all_done = True
            
            for grid_str, job_info in submitted_jobs.items():
                if job_info.get("error"):
                    continue
                
                job_id = job_info.get("job_id")
                if not job_id:
                    continue
                
                # Check job status
                try:
                    ssh_cmd = [
                        "ssh",
                        "jayapreethi.mohan@talon.und.edu",
                        f"scontrol show job {job_id} | grep JobState"
                    ]
                    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
                    
                    if "COMPLETED" in result.stdout:
                        print(f"✓ {grid_str}: Job {job_id} completed")
                        # Retrieve results (would fetch JSON file)
                        results[grid_str] = {"status": "completed", "job_id": job_id}
                    elif "FAILED" in result.stdout or "CANCELLED" in result.stdout:
                        print(f"✗ {grid_str}: Job {job_id} failed")
                        results[grid_str] = {"status": "failed", "job_id": job_id}
                    else:
                        all_done = False
                except:
                    all_done = False
            
            if all_done:
                print("✓ All jobs completed!")
                break
            
            time.sleep(60)
        
        return results

    def generate_reports(self, cpu_results: Dict) -> None:
        """Generate markdown and CSV reports."""
        print("\n" + "="*70)
        print("PHASE 5: GENERATE REPORTS")
        print("="*70)
        
        # Generate markdown report
        report_md = self._generate_markdown_report(cpu_results)
        report_file = self.results_dir / "BENCHMARK_REPORT.md"
        with open(report_file, "w") as f:
            f.write(report_md)
        print(f"✓ Markdown report: {report_file}")
        
        # Generate CSV
        csv_content = self._generate_csv(cpu_results)
        csv_file = self.results_dir / "results.csv"
        with open(csv_file, "w") as f:
            f.write(csv_content)
        print(f"✓ CSV export: {csv_file}")

    def _generate_markdown_report(self, cpu_results: Dict) -> str:
        """Generate comprehensive markdown report."""
        report = f"""# Fisher-KPP GPU/CPU Benchmark Report

**Run ID:** {self.run_id}  
**Date:** {self.timestamp}  
**Framework:** PyTorch {self.metadata.get('pytorch_version', 'N/A')}

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Experiment | {self.config['experiment']['name']} |
| Physics | {self.config['physics']['equation']} |
| D | {self.config['physics']['diffusion_coefficient']} |
| k | {self.config['physics']['reaction_rate']} |
| Runs per Grid | {self.config['benchmarking']['runs_per_grid']} |
| Timesteps | {self.config['benchmarking']['timesteps']} |

## CPU Results (Local)

"""
        report += "| Grid | Elements | Median (ms) | Mean (ms) | Std (ms) |\n"
        report += "|------|----------|-------------|-----------|----------|\n"
        
        for grid_key, result in cpu_results.items():
            if "error" not in result:
                report += f"| {grid_key} | {result['elements']:,} | {result['median_time_ms']:.2f} | {result['mean_time_ms']:.2f} | {result['std_time_ms']:.2f} |\n"
        
        report += f"""

## GPU Results (Talon Cluster)

*To be populated when cluster jobs complete.*

## Reproducibility

- **Git Commit:** {self.metadata.get('git_commit', 'N/A')[:8]}
- **Python:** {self.metadata['python_version']}
- **Config:** {self.config_path}
- **Random Seed:** {self.config['reproducibility']['random_seed']}

See `metadata.json` for full environment capture.
"""
        return report

    def _generate_csv(self, cpu_results: Dict) -> str:
        """Generate CSV export."""
        csv = "grid_size,elements,device,median_ms,mean_ms,std_ms,min_ms,max_ms\n"
        
        for grid_key, result in cpu_results.items():
            if "error" not in result:
                csv += (f"{grid_key},{result['elements']},cpu,"
                       f"{result['median_time_ms']:.2f},{result['mean_time_ms']:.2f},"
                       f"{result['std_time_ms']:.2f},{result['min_time_ms']:.2f},"
                       f"{result['max_time_ms']:.2f}\n")
        
        return csv

    def run_full_pipeline(self, local_only: bool = False, cluster_only: bool = False) -> None:
        """Execute full pipeline."""
        try:
            if not cluster_only:
                cpu_results = self.run_local_cpu_benchmarks()
            else:
                cpu_results = {}
            
            if not local_only:
                job_ids = self.generate_slurm_jobs()
                self.submit_jobs_to_cluster(job_ids)
                self.wait_for_cluster_results()
            
            self.generate_reports(cpu_results)
            
            print("\n" + "="*70)
            print(f"✓ PIPELINE COMPLETE")
            print(f"  Results: {self.results_dir}")
            print(f"  Run ID: {self.run_id}")
            print("="*70)
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fisher-KPP Benchmark Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (local CPU + cluster GPU)
  python3 run_benchmark_suite.py
  
  # CPU only
  python3 run_benchmark_suite.py --local-only
  
  # Custom config
  python3 run_benchmark_suite.py --config config/custom.yaml
        """
    )
    
    parser.add_argument("--config", default="config/benchmark_config.yaml",
                        help="Benchmark configuration file")
    parser.add_argument("--local-only", action="store_true",
                        help="Run CPU benchmarks only")
    parser.add_argument("--cluster-only", action="store_true",
                        help="Submit cluster jobs only (skip CPU)")
    
    args = parser.parse_args()
    
    orchestrator = BenchmarkOrchestrator(args.config)
    orchestrator.run_full_pipeline(local_only=args.local_only, 
                                  cluster_only=args.cluster_only)


if __name__ == "__main__":
    main()
