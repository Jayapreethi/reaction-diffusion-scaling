# Pipeline Quick Start

## 30-Second Tutorial

### Option 1: CPU Only (Fast, ~3-5 minutes)
```bash
make benchmark-cpu
```
Results → `outputs/benchmark_*/BENCHMARK_REPORT.md`

### Option 2: CPU + GPU Cluster
```bash
make benchmark
```
Automatically:
1. Runs CPU benchmarks locally
2. Submits GPU jobs to Talon
3. Waits for completion
4. Generates comprehensive reports

---

## Understanding the Output

After running, you'll find:

```
outputs/benchmark_20260716_abc123/
├── BENCHMARK_REPORT.md      ← Read this first!
├── results.csv              ← Import to Excel/Python
├── cpu_results.json         ← Raw CPU data
├── gpu_results_512x512.json ← Raw GPU data
└── metadata.json            ← Reproducibility info
```

### Example Report Section

```
| Grid | CPU (ms) | GPU (ms) | Speedup | Recommendation |
|------|----------|----------|---------|---|
| 128×128 | 28.3 | 105.6 | 0.27x | 🔴 CPU (faster) |
| 512×512 | 450 | 115 | 3.9x | 🟢 GPU |
```

---

## Configuration

Edit `config/benchmark_config.yaml` to customize:

```yaml
grids:
  - size: [256, 256]     # Which grid sizes to test
  - size: [512, 512]
  
benchmarking:
  runs_per_grid: 5       # How many times to run each
  timesteps: 50          # Simulation length

validation:
  conservation_check: true  # Physics validation
  max_l2_error: 1e-7        # GPU vs CPU tolerance
```

---

## Troubleshooting

### "PyYAML not found"
```bash
pip install -r requirements.txt
```

### CPU benchmark slow
This is expected for first run. Benchmarks take:
- 128×128: ~1 minute
- 256×256: ~2 minutes  
- 512×512: ~3 minutes
- 1024×1024: ~5 minutes

### GPU jobs not submitting
Verify cluster access:
```bash
ssh jayapreethi.mohan@talon.und.edu "echo OK"
```

---

## Common Workflows

### Run daily benchmarks for regression testing
```bash
# Cron job
0 9 * * * cd /path/to/reaction-diffusion && make benchmark-cpu
```

### Analyze multiple runs
```bash
# Compare CSV files across runs
paste <(cut -d, -f1-4 outputs/run1/results.csv) \
      <(cut -d, -f3-4 outputs/run2/results.csv)
```

### Generate custom report
```python
# Load and plot results
import pandas as pd
df = pd.read_csv("outputs/benchmark_*/results.csv", glob=True)
df.plot(x="grid_size", y="speedup", kind="bar")
```

---

## What's Being Measured

| Metric | Meaning |
|--------|---------|
| Median Time | Most representative single run (robust to outliers) |
| Speedup | CPU time / GPU time (>1 = GPU faster) |
| Conservation Residual | Mass conservation error (should be <1e-5) |
| L2 Error | GPU vs CPU numerical difference (should be <1e-7) |

---

## National Lab Standards

This pipeline implements NERSC reproducibility best practices:

✓ **Deterministic:** Same config → Same results  
✓ **Auditable:** Git commit captured with each run  
✓ **Documented:** Configuration, metadata, scripts all versioned  
✓ **Rigorous:** N=5 runs, proper CUDA sync, physics validation  
✓ **Portable:** Works on local machine and HPC clusters  

See `docs/PIPELINE_GUIDE.md` for full technical details.

---

## Next Steps

1. **Run CPU benchmark:** `make benchmark-cpu`
2. **Review results:** `cat outputs/benchmark_*/BENCHMARK_REPORT.md`
3. **Customize config:** Edit `config/benchmark_config.yaml`
4. **Submit to cluster:** `make benchmark`
5. **Analyze data:** Import `results.csv` to spreadsheet or Python

---

Questions? See:
- `docs/PIPELINE_GUIDE.md` - Full documentation
- `TECH_BLOG_POST.md` - Why GPU isn't always faster
- `docs/GPU_GUIDE.md` - GPU tuning and profiling
