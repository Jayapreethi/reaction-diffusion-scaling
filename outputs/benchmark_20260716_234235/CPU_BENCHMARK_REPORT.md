# CPU Benchmark Report

**Timestamp:** 20260716_234235

**Platform:** Windows-11-10.0.26200-SP0

**Python:** 3.12.10

**NumPy:** 1.26.4

## Performance Summary

| Grid Size | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | Stdev (ms) |
|-----------|-------------|-----------|----------|----------|------------|
| 128×128 | 5.95 | 6.79 | 5.33 | 9.23 | 1.57 |
| 256×256 | 161.51 | 147.93 | 117.90 | 167.27 | 20.23 |
| 512×512 | 887.42 | 948.82 | 878.47 | 1096.18 | 86.52 |
| 1024×1024 | 2961.61 | 3051.28 | 2797.02 | 3605.36 | 288.41 |

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

### 1024×1024

- **Initial Mass:** 0.09835434
- **Final Mass:** 0.52947506
- **Change:** 0.43112072

## Detailed Runs

### 128×128

- Run 1: 8.04 ms
- Run 2: 9.23 ms
- Run 3: 5.95 ms
- Run 4: 5.39 ms
- Run 5: 5.33 ms

### 256×256

- Run 1: 161.51 ms
- Run 2: 117.90 ms
- Run 3: 129.43 ms
- Run 4: 167.27 ms
- Run 5: 163.53 ms

### 512×512

- Run 1: 1096.18 ms
- Run 2: 887.42 ms
- Run 3: 878.47 ms
- Run 4: 999.64 ms
- Run 5: 882.37 ms

### 1024×1024

- Run 1: 3605.36 ms
- Run 2: 2961.61 ms
- Run 3: 3030.22 ms
- Run 4: 2797.02 ms
- Run 5: 2862.16 ms

