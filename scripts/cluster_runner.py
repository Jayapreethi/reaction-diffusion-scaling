#!/usr/bin/env python3
"""
Execute Benchmark Suite on Talon HPC Cluster
Simplified wrapper optimized for cluster execution via SSH

This script:
1. Copies project to cluster if needed
2. SSH into Talon
3. Runs benchmark pipeline remotely
4. Retrieves results back to local machine
"""

import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime


class ClusterBenchmarkRunner:
    """Execute benchmarks on Talon cluster."""
    
    def __init__(self, username: str = "jayapreethi.mohan", cluster: str = "talon.und.edu"):
        self.username = username
        self.cluster = cluster
        self.host = f"{username}@{cluster}"
        self.remote_project = "/home/jayapreethi.mohan/reaction_diffusion_scaling"
    
    def check_connectivity(self) -> bool:
        """Verify SSH connection to cluster."""
        try:
            result = subprocess.run(
                ["ssh", self.host, "echo OK"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def sync_project(self) -> bool:
        """Sync project to cluster."""
        print(f"📤 Syncing project to {self.cluster}...")
        try:
            subprocess.run(
                ["scp", "-r", ".", f"{self.host}:{self.remote_project}/"],
                check=True,
                capture_output=True,
                timeout=60
            )
            print("✓ Project synced")
            return True
        except Exception as e:
            print(f"✗ Sync failed: {e}")
            return False
    
    def run_cpu_benchmark(self) -> bool:
        """Run CPU benchmark on cluster."""
        print("\n📊 Running CPU benchmarks on cluster...")
        print(f"   Host: {self.cluster}")
        print(f"   Command: python3 scripts/run_benchmark_suite.py --local-only\n")
        
        try:
            subprocess.run([
                "ssh", self.host,
                f"cd {self.remote_project} && "
                f"python3 scripts/run_benchmark_suite.py --local-only "
                f"--config config/benchmark_config.yaml"
            ], check=True)
            print("✓ CPU benchmarking complete")
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False
    
    def retrieve_results(self) -> bool:
        """Retrieve results from cluster."""
        print("\n📥 Retrieving results...")
        
        # Find latest result directory on cluster
        try:
            result = subprocess.run([
                "ssh", self.host,
                f"ls -td {self.remote_project}/outputs/benchmark_*/ | head -1"
            ], capture_output=True, text=True, check=True)
            
            remote_dir = result.stdout.strip()
            if not remote_dir:
                print("✗ No results found on cluster")
                return False
            
            local_dir = Path("outputs")
            local_dir.mkdir(exist_ok=True)
            
            remote_basename = Path(remote_dir).name
            local_result_dir = local_dir / remote_basename
            
            # Copy results back
            subprocess.run([
                "scp", "-r",
                f"{self.host}:{remote_dir}",
                str(local_result_dir)
            ], check=True, capture_output=True)
            
            print(f"✓ Results retrieved: {local_result_dir}")
            return True
            
        except Exception as e:
            print(f"✗ Retrieval failed: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run full pipeline."""
        if not self.check_connectivity():
            print(f"❌ Cannot connect to {self.cluster}")
            print(f"   Try: ssh {self.host} echo OK")
            return False
        
        print(f"✓ Connected to {self.cluster}")
        
        if not self.sync_project():
            return False
        
        if not self.run_cpu_benchmark():
            return False
        
        if not self.retrieve_results():
            return False
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE")
        print("="*70)
        print("\nView results:")
        print("  Get-Content outputs/benchmark_*/BENCHMARK_REPORT.md")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Execute benchmarks on Talon HPC cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CPU benchmarks on Talon
  python scripts/cluster_runner.py
  
  # Check connectivity only
  python scripts/cluster_runner.py --check-only
        """
    )
    
    parser.add_argument("--username", default="jayapreethi.mohan",
                        help="Cluster username")
    parser.add_argument("--cluster", default="talon.und.edu",
                        help="Cluster hostname")
    parser.add_argument("--check-only", action="store_true",
                        help="Check connectivity only")
    
    args = parser.parse_args()
    
    runner = ClusterBenchmarkRunner(args.username, args.cluster)
    
    if args.check_only:
        if runner.check_connectivity():
            print(f"✓ Connected to {args.cluster}")
        else:
            print(f"✗ Cannot reach {args.cluster}")
        return
    
    success = runner.run_full_pipeline()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
