# Aggregate Benchmark Tables - Complete Guide

You now have **multiple ways** to view aggregate statistics from your CPU and GPU benchmarks!

---

## 1. CPU Benchmark Markdown Report

Each run creates `CPU_BENCHMARK_REPORT.md` with an **Aggregate Summary** section at the top:

```markdown
## Aggregate Summary

| Statistic | Value (ms) |
|-----------|-----------:|
| Average Median Time | 320.71 |
| Average Mean Time | 336.77 |
| Min Time (all runs) | 6.40 |
| Max Time (all runs) | 826.35 |
| Grid Sizes Tested | 3 |
| Total Runs | 15 |
```

**View with:** PowerShell, VS Code, or any text editor

```powershell
Get-Content outputs/benchmark_20260716_235221/CPU_BENCHMARK_REPORT.md
```

---

## 2. CPU Benchmark CSV Export

`CPU_BENCHMARK_RESULTS.csv` now includes an **Aggregate Summary** section at the top:

```
Aggregate Summary
Statistic,Value (ms)
Average Median Time,320.7072
Average Mean Time,336.7726
Min Time (all runs),6.4047
Max Time (all runs),826.3520
Grid Sizes Tested,3
Total Runs,15

Detailed Results
Grid Size,Median (ms),Mean (ms),Stdev (ms),...
128x128,6.4047,6.5640,0.9486
256x256,129.3648,139.0061,26.7657
...
```

**View with:** Excel, Google Sheets, or any spreadsheet app

**Import in Python:**
```python
import pandas as pd
df = pd.read_csv("CPU_BENCHMARK_RESULTS.csv", skiprows=7)
```

---

## 3. Benchmark Comparison Tool

Compare any two (or more) benchmark runs side-by-side using `scripts/compare_benchmarks.py`:

### List all available runs
```bash
python scripts/compare_benchmarks.py --list
```

Output:
```
Available Benchmark Runs:

1. 20260716_235221
   Grid sizes: 3, Average time: 320.71 ms
2. 20260716_234235
   Grid sizes: 4, Average time: 1004.12 ms
3. 20260716_234006
   Grid sizes: 4, Average time: 884.01 ms
```

### Compare two specific runs
```bash
python scripts/compare_benchmarks.py --run1 outputs/benchmark_20260716_235221 --run2 outputs/benchmark_20260716_234235
```

Output:
```
================================================================================
Benchmark Comparison: 20260716_235221 vs 20260716_234235
================================================================================

Grid Size       Run 1 (ms)      Run 2 (ms)      Difference      % Change  
--------------------------------------------------------------------------------
128×128                  6.40 ms          5.95 ms ↓         0.46 ms     -7.1%
256×256                129.36 ms        161.51 ms ↑        32.15 ms     24.8%
512×512                826.35 ms        887.42 ms ↑        61.07 ms      7.4%
--------------------------------------------------------------------------------
AVERAGE                320.71 ms       1004.12 ms         683.42 ms    213.1%
```

### Compare all recent runs
```bash
python scripts/compare_benchmarks.py --all
```

---

## 4. Aggregate Results (CPU vs GPU)

When combining CPU and GPU results using `scripts/aggregate_results.py`, includes:

```markdown
## Aggregate Summary

| Metric | CPU | GPU |
|--------|-----|-----|
| Average Time (ms) | 320.71 | 105.60 |
| Min Time (ms) | 6.40 | 95.30 |
| Max Time (ms) | 826.35 | 125.40 |
| Grid Sizes Tested | 3 | 3 |
| Average Speedup | — | 3.04x |
```

---

## Quick Reference

### Run CPU benchmark and view results

```powershell
# 1. Run benchmark
.\benchmark.ps1 benchmark-cpu

# 2. View markdown report (human-readable)
Get-Content outputs/benchmark_YYYYMMDD_HHMMSS/CPU_BENCHMARK_REPORT.md

# 3. Or view CSV in Excel
Invoke-Item outputs/benchmark_YYYYMMDD_HHMMSS/CPU_BENCHMARK_RESULTS.csv

# 4. Compare with previous run
python scripts/compare_benchmarks.py
```

### Output structure

```
outputs/
└── benchmark_20260716_235221/
    ├── CPU_BENCHMARK_REPORT.md      ← Markdown (aggregate + detailed)
    ├── CPU_BENCHMARK_RESULTS.csv    ← CSV (aggregate + detailed)
    └── cpu_results.json             ← Raw JSON data
```

---

## What These Aggregates Show

| Metric | Meaning |
|--------|---------|
| **Average Median Time** | Typical performance across all grid sizes |
| **Average Mean Time** | Mean performance (may be skewed by outliers) |
| **Min/Max Times** | Best and worst case scenarios |
| **Grid Sizes Tested** | Number of different problem sizes tested |
| **Total Runs** | N=5 runs per grid (statistical rigor) |
| **% Change** | Performance improvement/regression vs previous run |
| **Average Speedup** | Overall GPU vs CPU speedup |

---

## Use Cases

✅ **Publication-ready figures:** Copy markdown table directly into papers/reports  
✅ **Spreadsheet analysis:** Import CSV into Excel for custom charts/analysis  
✅ **Trend tracking:** Compare runs over time to monitor optimizations  
✅ **Performance review:** Quick overview of system behavior  
✅ **Regression testing:** Verify no performance degradation after code changes  
✅ **Optimization validation:** Show before/after comparison of improvements  

---

## Real Example

### Scenario: Testing optimization on 256×256 grid

**Before optimization:**
```
256×256: 140.5 ms (median), 142.3 ms (mean)
```

**After optimization:**
```
256×256: 120.3 ms (median), 121.8 ms (mean)
```

**Compare with tool:**
```bash
python scripts/compare_benchmarks.py --run1 outputs/benchmark_before --run2 outputs/benchmark_after
```

**Output shows:**
```
256×256    140.50 ms    120.30 ms ↓  20.20 ms    -14.4%
AVERAGE    ...          ...         ...         -14.4%
```

✓ **14.4% improvement** documented and verified!

---

## Key Features

- 🎯 **Automatic aggregation** - No manual calculation needed
- 📊 **Multiple formats** - Markdown, CSV, JSON, terminal output
- 📈 **Trend tracking** - Compare runs side-by-side
- 🔍 **Detailed breakdown** - Both aggregate and per-grid statistics
- 🧮 **Statistical rigor** - N=5 runs, median-based reporting
- 🎓 **Publication-ready** - Professional formatting for papers/reports

---

## Examples

### Example 1: View latest CPU benchmark report

```bash
# PowerShell
Get-Content outputs/benchmark_20260716_235221/CPU_BENCHMARK_REPORT.md

# Linux/Mac
cat outputs/benchmark_20260716_235221/CPU_BENCHMARK_REPORT.md
```

### Example 2: Compare last two runs

```bash
python scripts/compare_benchmarks.py
```

### Example 3: Export to Excel for further analysis

```bash
# Open CSV in Excel
Invoke-Item outputs/benchmark_20260716_235221/CPU_BENCHMARK_RESULTS.csv
```

### Example 4: Track performance over multiple optimization attempts

```bash
# Run 1
.\benchmark.ps1 benchmark-cpu
# ... make optimization ...

# Run 2
.\benchmark.ps1 benchmark-cpu
# ... make another optimization ...

# Compare all
python scripts/compare_benchmarks.py --all
```

---

**Ready to benchmark!** Just run `.\benchmark.ps1 benchmark-cpu` and check the aggregate tables. 🚀
