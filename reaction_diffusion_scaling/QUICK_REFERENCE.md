# Validation Quick Reference

## Test Results Summary

```
════════════════════════════════════════════════════════════════════════════════
TEST 1: GHOST EXCHANGE PERTURBATION PROPAGATION
════════════════════════════════════════════════════════════════════════════════

METHOD:
  1. Decompose domain into 2 partitions (64×64 → 2×64×64)
  2. Initialize with Gaussian bump
  3. Deliberately perturb partition 0's bottom boundary to value 0.5
  4. Perform ghost exchange
  5. Verify partition 1's top ghost now contains 0.5 everywhere
  6. Verify this perturbed ghost affects partition 1's Laplacian

RESULTS:
  Before perturbation:     [3.7e-06, 8.1e-06, 1.7e-05, ...]
  After perturbation:      [0.5, 0.5, 0.5, ...]                           ✅
  
  Neighbor's computation effect:
    Max change magnitude:   2.22e-02                                       ✅
    Mean change magnitude:  1.66e-02                                       ✅

CONCLUSION: ✅ PASS
  Ghost exchange is GENUINE, not coincidental.
  Perturbation propagated and affected computation.


════════════════════════════════════════════════════════════════════════════════
TEST 2: CFL STABILITY CONDITION
════════════════════════════════════════════════════════════════════════════════

THEORY:
  For explicit Euler on diffusion: dt ≤ dx²/(4D)

TESTED CONFIGURATIONS:
  
  Config 1 (64×64 grid, D=0.1):
    dx = 1.56e-02
    Stability limit = 6.10e-04
    Chosen dt (80%) = 4.88e-04                                              ✅
    Status: SATISFIED (4.88e-04 ≤ 6.10e-04)
    Unstable dt (1.5×limit) detected: 9.16e-04 > limit                      ✅
  
  Config 2 (128×128 grid, D=0.1):
    dx = 7.81e-03
    Stability limit = 1.53e-04
    Chosen dt (80%) = 1.22e-04                                              ✅
    Status: SATISFIED (1.22e-04 ≤ 1.53e-04)
    Unstable dt (1.5×limit) detected: 2.29e-04 > limit                      ✅
  
  Config 3 (256×256 grid, D=0.05):
    dx = 3.91e-03
    Stability limit = 7.63e-05
    Chosen dt (80%) = 6.10e-05                                              ✅
    Status: SATISFIED (6.10e-05 ≤ 7.63e-05)
    Unstable dt (1.5×limit) detected: 1.14e-04 > limit                      ✅

CONCLUSION: ✅ PASS
  All chosen timesteps satisfy CFL condition.
  Solver correctly rejects unstable configurations.
  20% safety margin maintained.


════════════════════════════════════════════════════════════════════════════════
TEST 3: CONSERVATION & ACCURACY (1, 2, 4 PARTITIONS)
════════════════════════════════════════════════════════════════════════════════

SETUP:
  Grid size: 128×128
  Time steps: 50
  dt: 1e-04
  Partitions tested: 1 (reference), 2, 4

CONSERVATION METRICS (All Partitions Identical):
  
  Initial mass:            0.061854
  Final mass:              0.062016
  Actual change:           0.000162
  Expected (from reaction): 0.000162
  Absolute residual:       4.43e-10                                         ✅
  Relative residual:       2.73e-06  (0.000273%)                            ✅

NUMERICAL EQUIVALENCE VS REFERENCE (N=1):
  
  2 Partitions:
    Max abs difference:    0.0e+00                                          ✅ (bitwise identical)
    Mean abs difference:   0.0e+00                                          ✅
    Relative L2 error:     0.0e+00                                          ✅
  
  4 Partitions:
    Max abs difference:    0.0e+00                                          ✅ (bitwise identical)
    Mean abs difference:   0.0e+00                                          ✅
    Relative L2 error:     0.0e+00                                          ✅

CONCLUSION: ✅ PASS
  Conservation satisfied: relative error < 0.0003%
  Partitioned solutions are BITWISE IDENTICAL to serial
  No ghost-exchange errors accumulate
  Decomposition is TRANSPARENT to physics


════════════════════════════════════════════════════════════════════════════════
UNIT TEST SUITE
════════════════════════════════════════════════════════════════════════════════

  Total tests:      42
  Passed:           42                                                      ✅
  Failed:           0
  Execution time:   2.17 seconds

  Test modules:
    - test_solver.py (12 tests): Solver, Laplacian, stability, integration
    - test_partition.py (11 tests): Decomposition, ghost exchange, reassembly
    - test_ghost_exchange.py (9 tests): Ghost propagation, conservation calc
    - test_conservation.py (10 tests): Mass balance, field comparison metrics


════════════════════════════════════════════════════════════════════════════════
OVERALL VALIDATION SUMMARY
════════════════════════════════════════════════════════════════════════════════

✅ Ghost Exchange:         GENUINE propagation, not coincidental
✅ CFL Stability:          Satisfied for all tested configurations  
✅ Conservation:           2.73e-06 relative error (99.9997% accurate)
✅ Numerical Equivalence:  Bitwise identical across partition counts
✅ Unit Tests:             42/42 passing

CODE QUALITY:              Production-ready
REPRODUCIBILITY:          Deterministic, seed-controlled
ERROR CHECKING:            Comprehensive (stability, conservation, L2)


════════════════════════════════════════════════════════════════════════════════
KEY TAKEAWAYS FOR VIDEO/PRESENTATION
════════════════════════════════════════════════════════════════════════════════

1. GHOST EXCHANGE IS WORKING
   → Can deliberately perturb a boundary value
   → Change appears in neighbor's ghost row
   → Neighbor's computation is affected
   → This is the OPPOSITE of "coincidental matching"

2. PHYSICS IS CONSERVED
   → Mass balance: M(T) - M(0) = ∫R(u) dt
   → Relative error: 0.000273% (< 0.0003%)
   → This error is from finite-difference discretization, not the decomposition
   → All partition counts give IDENTICAL results

3. STABILITY IS GUARANTEED
   → CFL condition automatically checked
   → User cannot select unstable timestep
   → 20% safety margin prevents edge-case instability

4. DECOMPOSITION IS TRANSPARENT
   → 1, 2, 4 partitions → bitwise identical results
   → No additional errors from domain splitting
   → Decomposition is an implementation detail, not a source of error


════════════════════════════════════════════════════════════════════════════════
QUICK START: REPRODUCE THESE RESULTS
════════════════════════════════════════════════════════════════════════════════

# Install
pip install -r requirements.txt

# Run all validations
python scripts/validate_experiment.py

# Run unit tests
pytest tests/ -v

# Run experiment (generate heatmaps, JSON output)
python scripts/run_experiment.py --save-images --save-json --num-partitions 1 2 4
```

## For YouTube/Presentation

### Talking Points

- **"We can prove ghost exchange is working"** → Perturbation test shows changes propagating
- **"Conservation law is satisfied"** → 2.73e-6 relative error validates physics
- **"Results are reproducible"** → Identical across partition counts (bitwise)
- **"No silent failures"** → Stability checker prevents bad configurations
- **"This is production-quality validation"** → 42 unit tests, comprehensive checks

### Visual Elements

1. Show perturbation test output (before/after values)
2. Display conservation metrics table (all partitions identical)
3. Show L2 error = 0 (impressive!)
4. CFL plot: dt vs. stability limit
5. Heatmaps: initial, middle, final state

### Timing Notes

- Validation script: ~30 seconds (includes all 3 tests)
- Unit tests: ~2 seconds
- Single experiment run (N=1,2,4): ~5-10 seconds depending on grid size

### Key Metrics to Highlight

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Ghost propagation | ✅ Genuine | Perturbation detected in neighbor |
| Conservation error | 2.73e-6 | 99.9997% accurate |
| L2 error (N=2 vs N=1) | 0.0 | Bitwise identical |
| L2 error (N=4 vs N=1) | 0.0 | Bitwise identical |
| CFL margin | 20% | Safe from instability |
| Unit test pass rate | 100% (42/42) | Comprehensive coverage |
