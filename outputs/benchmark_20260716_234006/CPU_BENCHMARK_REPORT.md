# CPU Benchmark Report

**Timestamp:** 20260716_234006

**Platform:** Windows-11-10.0.26200-SP0

**Python:** 3.12.10

**NumPy:** 1.26.4

## Performance Summary

| Grid Size | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | Stdev (ms) |
|-----------|-------------|-----------|----------|----------|------------|
| 128×128 | 6.24 | 6.17 | 5.44 | 7.22 | 0.67 |
| 256×256 | 113.86 | 116.34 | 101.21 | 140.40 | 13.42 |
| 512×512 | 675.03 | 712.44 | 662.54 | 857.53 | 73.63 |
| 1024×1024 | 2740.93 | 2776.53 | 2722.15 | 2877.25 | 61.93 |

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

- Run 1: 5.46 ms
- Run 2: 5.44 ms
- Run 3: 7.22 ms
- Run 4: 6.24 ms
- Run 5: 6.49 ms

### 256×256

- Run 1: 113.86 ms
- Run 2: 140.40 ms
- Run 3: 118.84 ms
- Run 4: 107.40 ms
- Run 5: 101.21 ms

### 512×512

- Run 1: 857.53 ms
- Run 2: 675.03 ms
- Run 3: 662.54 ms
- Run 4: 667.81 ms
- Run 5: 699.27 ms

### 1024×1024

- Run 1: 2722.42 ms
- Run 2: 2877.25 ms
- Run 3: 2722.15 ms
- Run 4: 2740.93 ms
- Run 5: 2819.89 ms

