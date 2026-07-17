# CPU Benchmark Report

**Timestamp:** 20260716_235221

**Platform:** Windows-11-10.0.26200-SP0

**Python:** 3.12.10

**NumPy:** 1.26.4

## Aggregate Summary

| Statistic | Value (ms) |
|-----------|-----------:|
| Average Median Time | 320.71 |
| Average Mean Time | 336.77 |
| Min Time (all runs) | 6.40 |
| Max Time (all runs) | 826.35 |
| Grid Sizes Tested | 3 |
| Total Runs | 15 |

## Performance Summary

| Grid Size | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | Stdev (ms) |
|-----------|-------------|-----------|----------|----------|------------|
| 128×128 | 6.40 | 6.56 | 5.42 | 7.96 | 0.95 |
| 256×256 | 129.36 | 139.01 | 119.19 | 191.91 | 26.77 |
| 512×512 | 826.35 | 864.75 | 720.23 | 1051.93 | 133.49 |

## Physics Validation

### 128×128

- **Initial Mass:** 0.09971421
- **Final Mass:** 0.51466303
- **Change:** 0.41494882

### 256×256

- **Initial Mass:** 0.09893373
- **Final Mass:** 0.51938485
- **Change:** 0.42045113

### 512×512

- **Initial Mass:** 0.09854690
- **Final Mass:** 0.47792786
- **Change:** 0.37938096

## Detailed Runs

### 128×128

- Run 1: 6.40 ms
- Run 2: 7.96 ms
- Run 3: 7.30 ms
- Run 4: 5.42 ms
- Run 5: 5.74 ms

### 256×256

- Run 1: 129.36 ms
- Run 2: 191.91 ms
- Run 3: 123.92 ms
- Run 4: 119.19 ms
- Run 5: 130.65 ms

### 512×512

- Run 1: 988.50 ms
- Run 2: 1051.93 ms
- Run 3: 736.72 ms
- Run 4: 720.23 ms
- Run 5: 826.35 ms

