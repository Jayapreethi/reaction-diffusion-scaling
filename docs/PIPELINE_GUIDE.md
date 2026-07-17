# Reproducible Benchmark Pipeline

## Overview

The Fisher-KPP GPU/CPU benchmark pipeline follows **national lab standards for reproducible computational science**. It automates:

1. **CPU benchmarking** on your local machine
2. **GPU job submission** to Talon HPC cluster  
3. **Result aggregation** across platforms
4. **Report generation** in multiple formats (JSON, CSV, markdown)

This design ensures complete reproducibility: same configuration → same results, every time.

---

## Quick Start

### CPU Benchmarking Only (Fast, Local)
```bash
# Run benchmarks on your machine (takes ~2-5 minutes)
make benchmark-cpu

# Results in: outputs/benchmark_YYYYMMDD_HHMMSS/
```

### Full Pipeline (CPU + GPU Cluster)
```bash
# Requires SSH access to talon.und.edu
make benchmark

# Monitors Talon job queue and retrieves results when complete
```

### View Results
```bash
# Aggregate and generate reports for latest run
make results

# Open report
open outputs/benchmark_*/BENCHMARK_REPORT.md
```

---

## Architecture

### Three-Layer Design

```
LOCAL MACHINE                    CLUSTER (TALON)
┌─────────────────────┐         ┌──────────────────┐
│  Master Orchestrator│◄────────│  Job Status API  │
│  (Python)           │         └──────────────────┘
│                     │         
│ • Reads config      │         ┌──────────────────┐
│ • Runs CPU bench    │         │  SLURM GPU Jobs  │
│ • Generates jobs    │────────►│  (Bash scripts)  │
│ • Polls status      │         │                  │
│ • Collects results  │◄────────│  • Benchmark     │
│ • Aggregates data   │         │  • Validation    │
│ • Reports          │         │  • JSON output   │
└─────────────────────┘         └──────────────────┘

Result Flow:
  CPU JSON ──┐
             ├──► Aggregator ──► BENCHMARK_REPORT.md
  GPU JSON ──┤                ──► results.csv
             └──► Speedup analysis
```

### Key Components

#### 1. **Configuration** (`config/benchmark_config.yaml`)
- **Single source of truth** for all experiment parameters
- Defines: physics parameters, grid sizes, resource allocation, validation thresholds
- Version controlled for audit trail
- National lab standard: all configurable parameters in one file

#### 2. **Master Orchestrator** (`scripts/run_benchmark_suite.py`)
- Python CLI with subcommands
- Captures environment (Python version, PyTorch, CUDA, git state)
- Runs CPU benchmarks locally with proper CUDA synchronization
- Generates reproducible SLURM job scripts
- Polls cluster for job completion
- Aggregates results into unified output

#### 3. **GPU Benchmark** (`scripts/gpu_benchmark_single.py`)
- Single-grid-size benchmark for cluster execution
- Proper CUDA synchronization before/after timing
- Optional physics validation (mass conservation checks)
- JSON output for machine processing

#### 4. **Result Aggregation** (`scripts/aggregate_results.py`)
- Combines CPU and GPU results
- Computes speedups and crossover analysis
- Generates multiple report formats
- Enables data-driven decision making

#### 5. **Makefile** (`Makefile`)
- Simple interface for common operations
- Documents available targets
- National lab standard: reproducible via `make` commands

---

## Configuration Format

National lab standard configuration in YAML:

```yaml
experiment:
  name: "Fisher-KPP GPU vs CPU Scaling Study"
  version: "1.0"
  date: "2026-07-16"

physics:
  diffusion_coefficient: 0.1
  reaction_rate: 0.5
  timestep: 0.01

grids:
  - size: [128, 128]  # Small: CPU wins
  - size: [512, 512]  # Medium: GPU wins  
  - size: [1024, 1024] # Large: GPU dominates

benchmarking:
  runs_per_grid: 5  # Statistical rigor
  statistical_metric: "median"  # Robust to outliers

validation:
  conservation_check: true
  tolerance_conservation: 1e-5
  max_l2_error: 1e-7

resources:
  slurm:
    partition: "talon-gpu32"
    time_limit: "00:30:00"
    gpu_type: "v100"

reproducibility:
  random_seed: 42  # Same initial conditions every run
  git_commit: "auto"
  environment_capture: true
```

**Why this matters:**
- Parameter audit trail
- Easy comparison across runs
- Version control friendly
- Documentation of methodology

---

## Workflow

### 1. CPU Benchmarking (Local, ~2-5 minutes)

```python
# Master orchestrator loads config
config = load_yaml("config/benchmark_config.yaml")

# For each grid size (128×128, 256×256, 512×512, 1024×1024):
#   - Initialize solver on CPU
#   - Run N=5 times with proper synchronization
#   - Record median, mean, std deviation
#   - Save to cpu_results.json
```

**Output:** `outputs/benchmark_YYYYMMDD_HHMMSS/cpu_results.json`

```json
{
  "512x512": {
    "elements": 262144,
    "median_time_ms": 115.42,
    "mean_time_ms": 116.89,
    "std_time_ms": 2.15,
    "run_times_ms": [115.42, 114.89, 118.92, 116.34, 115.67]
  }
}
```

### 2. GPU Job Generation (Seconds)

For each grid size, orchestrator generates a SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=fisher-kpp-512x512
#SBATCH --partition=talon-gpu32
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

python3 scripts/gpu_benchmark_single.py \
    --grid-size 512 512 \
    --timesteps 50 \
    --runs 5 \
    --seed 42 \
    --output gpu_results_512x512.json \
    --validate
```

**National lab standard:**
- Deterministic job scripts (same input → same script)
- CUDA synchronization for accurate timing
- Physics validation built-in
- Version-controlled parameters

### 3. Job Submission & Polling (Via SSH)

```python
# For each job script:
#   1. SCP to cluster
#   2. SSH sbatch to submit
#   3. Parse job ID from SLURM response
#   4. Poll job status every 60 seconds
#   5. When completed, SCP results back to local
```

**Status tracking:** `outputs/benchmark_*/submitted_jobs.json`

```json
{
  "512x512": {
    "job_id": "267451",
    "status": "completed",
    "submitted_time": "2026-07-16T14:30:45"
  }
}
```

### 4. Result Aggregation

Combines all results (CPU + GPU) and computes:
- Performance metrics (median, mean, std)
- GPU vs CPU speedup
- Crossover analysis (where GPU becomes faster)
- Physics validation status

**Output files:**
- `BENCHMARK_REPORT.md` - Human-readable summary with tables
- `results.csv` - Tabular data for spreadsheets
- `speedups.json` - Speedup analysis
- `metadata.json` - Environment snapshot

---

## Output Structure

```
outputs/
└── benchmark_20260716_abc123ef/  ← Timestamped run directory
    ├── metadata.json              ← Environment capture
    ├── cpu_results.json           ← Local CPU benchmark
    ├── gpu_results_128x128.json   ← Cluster GPU result
    ├── gpu_results_256x256.json
    ├── gpu_results_512x512.json
    ├── gpu_results_1024x1024.json
    ├── BENCHMARK_REPORT.md        ← Main report
    ├── results.csv                ← For analysis
    ├── speedups.json              ← Speedup summary
    ├── submitted_jobs.json        ← Job tracking
    └── slurm_jobs/                ← Generated scripts
        ├── job_128x128.sh
        ├── job_256x256.sh
        └── ...
```

---

## National Lab Standards Compliance

This pipeline follows DOE/NERSC/LLNL best practices:

### ✓ Reproducibility
- Single configuration file controls all parameters
- Random seed (42) ensures identical initial conditions
- Git commit hash captured with each run
- Full environment metadata logged

### ✓ Automation
- Single command to run entire pipeline
- No manual job submission
- Automatic result retrieval
- No copy-paste errors

### ✓ Rigor
- Multiple runs (N=5) per configuration
- Proper CUDA synchronization (avoids timing bugs)
- Physics validation (conservation, accuracy)
- Statistical analysis (median, not mean)

### ✓ Traceability
- Run ID uniquely identifies each execution
- All files timestamped
- Configuration versioned in git
- Results linked to git commit

### ✓ Portability
- Works on local machine or HPC cluster
- SSH-based (no special tools required)
- SLURM standard (works on any SLURM cluster)
- Python with PyTorch (portable across systems)

---

## Usage Examples

### Run CPU benchmarks only (fast, testing)
```bash
make benchmark-cpu
```

### Submit to cluster without local CPU run
```bash
make benchmark-cluster
```

### Full pipeline (CPU + GPU)
```bash
make benchmark
# Monitor progress in real-time
# Results aggregated automatically when cluster jobs complete
```

### Check pipeline status
```bash
make status
```

### View latest report
```bash
ls -t outputs/benchmark_*/BENCHMARK_REPORT.md | head -1 | xargs cat
```

### Compare multiple runs
```bash
# Results in CSV format for Excel/Python analysis
cat outputs/benchmark_*/results.csv | column -t -s,
```

### Clean old results
```bash
make clean
```

---

## Troubleshooting

### CPU benchmark fails
```bash
# Check PyTorch installation
python3 -c "import torch; print(torch.cuda.is_available())"

# Verify reaction_diffusion module
python3 -c "from reaction_diffusion.solver import FisherKPPSolver"
```

### SLURM job submission fails
```bash
# Verify SSH access to Talon
ssh jayapreethi.mohan@talon.und.edu "echo OK"

# Check SLURM availability
ssh jayapreethi.mohan@talon.und.edu "sinfo"
```

### Results missing after cluster job completes
```bash
# Check job output manually
ssh jayapreethi.mohan@talon.und.edu \
  "ls /home/jayapreethi.mohan/reaction_diffusion_scaling/outputs/"

# Manually retrieve results
scp jayapreethi.mohan@talon:/home/jayapreethi.mohan/reaction_diffusion_scaling/outputs/gpu_results_*.json \
  outputs/
```

---

## Future Enhancements

- [ ] Email notifications when jobs complete
- [ ] Multi-GPU benchmarking
- [ ] Time-series monitoring (daily runs)
- [ ] Automated performance regression detection
- [ ] Integration with Weights & Biases for experiment tracking
- [ ] Multi-cluster support (Talon, Medora, cloud)

---

## References

- **National Labs:** [NERSC Reproducibility](https://docs.nersc.gov/)
- **Configuration:** YAML format per [PyYAML](https://pyyaml.org/)
- **HPC:** SLURM Job Scheduler [documentation](https://slurm.schedmd.com/)
- **Physics:** [Fisher-KPP Equation](https://en.wikipedia.org/wiki/Fisher%27s_equation)
