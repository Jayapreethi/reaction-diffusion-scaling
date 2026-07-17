# Windows Setup & Execution Guide

## Problem: PyTorch on Windows

The local Windows PyTorch installation is broken (missing `__version__` attribute). This is common with incomplete wheel downloads on unstable networks.

**Solution:** Run benchmarks on **Talon HPC cluster** instead, where PyTorch is properly configured.

---

## Quick Start (Recommended: Use Talon)

### Option 1: Run Benchmarks on Talon via SSH (Easiest)

```powershell
# 1. Check SSH connectivity
ssh jayapreethi.mohan@talon.und.edu echo OK

# 2. Run benchmarks on cluster
python scripts/cluster_runner.py

# 3. Results automatically retrieved to local machine
Get-Content outputs/benchmark_*/BENCHMARK_REPORT.md
```

**What happens:**
- Syncs project to cluster via SCP
- SSH runs benchmark pipeline remotely
- Automatically copies results back
- All in one command!

### Option 2: Manual Cluster Execution

```powershell
# SSH into cluster
ssh jayapreethi.mohan@talon.und.edu

# On cluster, run benchmark
cd /home/jayapreethi.mohan/reaction_diffusion_scaling
python3 scripts/run_benchmark_suite.py --local-only --config config/benchmark_config.yaml

# Back on local machine, retrieve results
scp -r jayapreethi.mohan@talon.und.edu:/home/jayapreethi.mohan/reaction_diffusion_scaling/outputs/benchmark_* ./outputs/
```

---

## Windows PowerShell Wrapper

For convenience, use the PowerShell wrapper (Windows equivalent of `make`):

```powershell
# Run CPU benchmarks (on cluster via automation)
.\benchmark.ps1 benchmark-cpu

# Check documentation
.\benchmark.ps1 docs

# View results
.\benchmark.ps1 results

# Check status of latest runs
.\benchmark.ps1 status

# Clean old results
.\benchmark.ps1 clean
```

---

## Fix: Local PyTorch on Windows (Optional)

If you want to fix local PyTorch:

```powershell
# Remove broken installation
pip uninstall torch -y

# Reinstall with stable wheel
pip install --upgrade --force-reinstall torch numpy

# Verify
python -c "import torch; print(torch.__version__)"
```

If installation still fails, it's likely a network issue. Use cluster instead.

---

## File Structure

```
reaction-diffusion/
├── benchmark.ps1                  ← Use this on Windows!
├── scripts/
│   ├── run_benchmark_suite.py    ← Main orchestrator
│   ├── cluster_runner.py         ← SSH execution (NEW)
│   ├── gpu_benchmark_single.py   ← Talon execution
│   └── aggregate_results.py      ← Result processing
├── config/
│   └── benchmark_config.yaml     ← All parameters
└── outputs/
    └── benchmark_*/              ← Results (auto-retrieved)
```

---

## Recommended Workflow

### 1. Initial Setup (One-time)

```powershell
# Verify SSH access
ssh jayapreethi.mohan@talon.und.edu "echo OK"

# Test Python on cluster
ssh jayapreethi.mohan@talon.und.edu "python3 --version"
```

### 2. Run Benchmarks

```powershell
# Execute on cluster (easiest)
python scripts/cluster_runner.py

# Or use PowerShell wrapper
.\benchmark.ps1 benchmark-cpu
```

### 3. View Results

```powershell
# Check latest report
Get-Content outputs/benchmark_*/BENCHMARK_REPORT.md -Head 100

# View CSV for analysis
Import-Csv outputs/benchmark_*/results.csv | Format-Table

# Check all runs
.\benchmark.ps1 status
```

---

## Troubleshooting

### SSH Connection Fails
```powershell
# Test connectivity
ssh jayapreethi.mohan@talon.und.edu "sinfo | head -5"

# If blocked, check:
# - VPN connected (if required)
# - Firewall allows SSH (port 22)
# - SSH key configured
```

### PyTorch Import Error on Cluster

```powershell
# SSH to cluster and debug
ssh jayapreethi.mohan@talon.und.edu
python3 -c "import torch; print(torch.__version__)"

# If fails, module might need loading:
module load python/3.11 pytorch  # varies by cluster config
```

### Results Not Retrieved

```powershell
# Check results exist on cluster
ssh jayapreethi.mohan@talon.und.edu "ls -la /home/jayapreethi.mohan/reaction_diffusion_scaling/outputs/"

# Manually copy
scp -r jayapreethi.mohan@talon.und.edu:/home/jayapreethi.mohan/reaction_diffusion_scaling/outputs/benchmark_* ./outputs/
```

---

## Alternative: Use Git Bash / WSL

If you want `make` to work on Windows:

### Option A: Git Bash
```bash
# Install Git for Windows (includes bash)
# Then use in Git Bash terminal:
make benchmark-cpu
```

### Option B: Windows Subsystem for Linux (WSL)
```powershell
# In WSL terminal:
make benchmark-cpu
```

---

## Comparison: Local vs Cluster

| Aspect | Local Windows | Talon Cluster |
|--------|---------------|---------------|
| PyTorch Status | ❌ Broken | ✅ Working |
| GPU Available | ❌ No | ✅ Tesla V100 |
| Python Version | System | 3.11+ optimized |
| Execution Time | N/A | ~5-10 min |
| Recommended | ❌ | ✅ YES |

**Bottom Line:** Run on **Talon cluster**. It's faster, more reliable, and you get GPU results too.

---

## Next Steps

1. **Test connectivity:**
   ```powershell
   ssh jayapreethi.mohan@talon.und.edu "echo OK"
   ```

2. **Run benchmarks:**
   ```powershell
   python scripts/cluster_runner.py
   ```

3. **View results:**
   ```powershell
   Get-Content outputs/benchmark_*/BENCHMARK_REPORT.md -Head 50
   ```

4. **Full documentation:**
   ```powershell
   Get-Content docs/PIPELINE_GUIDE.md | more
   ```

---

**Questions?** See `PIPELINE_QUICKSTART.md` or `docs/PIPELINE_GUIDE.md` for detailed information.
