# Fisher–KPP Reaction–Diffusion Scaling Experiment

A standalone scientific-computing demonstration of domain decomposition in the two-dimensional Fisher–KPP reaction–diffusion equation.

## Research Question

**When a scientific simulation is decomposed across computational partitions, how do we verify that it remains physically and numerically correct?**

This project demonstrates:
- How physical correctness is verified through mass conservation.
- How numerical results are compared across decomposition strategies.
- How ghost-cell exchange mechanisms work and why they matter for simulation fidelity.
- The distinction between single-process partitioning (this experiment) and true parallel scaling (future work).

## The Fisher–KPP Equation

The Fisher–KPP (Fisher–Kolmogorov–Petrovsky–Piskunov) reaction–diffusion equation models spatial spread of a population with logistic growth:

$$\frac{\partial u}{\partial t} = D\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) + ku(1-u)$$

### Components

- **Diffusion term** ($D \Delta u$): Models dispersal or spreading.
- **Reaction term** ($ku(1-u)$): Models logistic population growth; saturates at $u=1$.
- **Boundary conditions**: Zero-flux (Neumann) on all four boundaries; no material crosses the domain boundary.

### Physical Interpretation

- $u(x,y,t) \in [0,1]$ represents population density or normalized concentration.
- A population starting from a localized perturbation spreads outward.
- The diffusive and reactive timescales compete to determine the propagation speed.

## Numerical Method

### Spatial Discretization

- **Grid**: Uniform square grid with spacing $\Delta x = \Delta y$.
- **Laplacian**: Standard five-point finite-difference stencil:
  $$\Delta u_{i,j} = \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{\Delta x^2}$$
- **Boundary conditions**: Zero-flux (Neumann) implemented by reflecting ghosts.

### Time Integration

- **Method**: Explicit Euler.
- **Update rule**:
  $$u_{i,j}^{n+1} = u_{i,j}^n + \Delta t \left[ D \Delta u_{i,j}^n + k u_{i,j}^n (1 - u_{i,j}^n) \right]$$
- **Stability constraint**: $\Delta t \leq \frac{\Delta x^2}{4D}$ (CFL condition for parabolic problems).

## Strip-Based Domain Decomposition

The 2D grid is partitioned into **horizontal strips** by splitting only along the row dimension.

### Partition Structure

Each partition maintains:
- **Owned rows**: The rows assigned to this partition.
- **Top ghost row** (if not the first partition): Copy of the bottom row of the upper neighbor.
- **Bottom ghost row** (if not the last partition): Copy of the top row of the lower neighbor.

### Ghost Exchange Algorithm

At every time step:

1. **Synchronize boundaries**:
   - Each partition sends its bottom boundary row to the partition below.
   - Each partition sends its top boundary row to the partition above.
   - Receive and populate ghost rows.

2. **Local update**:
   - Compute Laplacian and reaction on all rows (owned + ghosts).
   - Update only owned rows.
   - Ghost rows remain from synchronization step.

3. **Repeat**: Step 1 for the next time step.

### Handling Uneven Decomposition

When the number of rows is not evenly divisible by the number of partitions, early partitions receive extra rows to balance the load.

## Conservation Validation

### Mass Balance (Continuous)

With zero-flux boundaries, the diffusion term does not change total mass. Therefore:

$$M(T) - M(0) = \int_0^T \int_{\Omega} R(u) \, d\Omega \, dt = \int_0^T \int_{\Omega} k u(1-u) \, d\Omega \, dt$$

### Numerical Validation

1. Compute initial mass: $M(0) = \Delta x^2 \sum_{i,j} u_{i,j}^0$.
2. Accumulate reaction integral over time steps: $\sum_n R(u^n) \cdot \Delta t \cdot \Delta x^2$.
3. Compute final mass: $M(T) = \Delta x^2 \sum_{i,j} u_{i,j}^T$.
4. Check conservation residual: $|M(T) - M(0) - \text{accumulated reaction}|$.

### Reported Metrics

- **Absolute residual**: $| \Delta M_{\text{actual}} - \Delta M_{\text{expected}} |$.
- **Relative residual**: $| \Delta M_{\text{actual}} - \Delta M_{\text{expected}} | / |\Delta M_{\text{expected}}|$.

Small residuals indicate physical correctness despite discretization errors.

## Numerical Equivalence Testing

Compare partitioned solutions against a reference (single-partition) solution:

- **Maximum absolute difference**: $\max_{i,j} |u^{\text{part}}_{i,j} - u^{\text{ref}}_{i,j}|$.
- **Mean absolute difference**: Average absolute difference.
- **Relative L₂ error**: $\frac{\|u^{\text{part}} - u^{\text{ref}}\|_2}{\|u^{\text{ref}}\|_2}$.

Results from different partition counts should agree within floating-point tolerance when ghost exchange is correctly implemented.

## Project Structure

```
reaction_diffusion_scaling/
├── README.md                          # This file
├── VALIDATION_REPORT.md               # Comprehensive validation results
├── requirements.txt                   # Python dependencies
├── reaction_diffusion/
│   ├── __init__.py
│   ├── solver.py                      # Core finite-difference solver
│   ├── partition.py                   # Domain decomposition and ghost exchange
│   ├── conservation.py                # Conservation validation
│   └── visualize.py                   # Heatmap visualization
├── scripts/
│   ├── run_experiment.py              # Main experiment runner (CLI)
│   └── validate_experiment.py         # Validation suite (ghost exchange, CFL, conservation)
├── tests/
│   ├── test_solver.py
│   ├── test_partition.py
│   ├── test_ghost_exchange.py
│   └── test_conservation.py
└── outputs/                           # Results and images (generated)
```

## Installation

1. Clone or download this project.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Validation & Verification

Before running large experiments, verify that the implementation is correct using the comprehensive validation suite.

### Quick Validation

Run all validation tests (ghost exchange, CFL stability, conservation):

```bash
python scripts/validate_experiment.py
```

This test suite verifies:

1. **Ghost Exchange Propagation** — Perturb a partition boundary and confirm the change propagates through ghost cells to the neighbor, and affects that neighbor's computation.
2. **CFL Stability** — Confirm the chosen timestep satisfies the explicit scheme stability condition: $\Delta t \leq \frac{\Delta x^2}{4D}$.
3. **Conservation & Accuracy** — Run with 1, 2, and 4 partitions; verify mass balance is satisfied and partitioned results match the serial reference (bitwise identical).

### Expected Output

```
✅ Ghost Exchange                 PASS
✅ CFL Stability                  PASS
✅ Conservation Consistency       PASS

Max conservation residual:        2.73e-06 (< 0.0003%)
Max L2 error vs reference:        0.00e+00 (floating-point precision)
```

### Detailed Validation Report

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for:
- In-depth methodology for each test
- Detailed numerical results
- Physical interpretation
- Recommendations for extensions

## Running the Experiment

### Basic Usage

Run the experiment with default parameters (CPU, 128×128 grid, 100 time steps, 1/2/4 partitions):

```bash
python scripts/run_experiment.py
```

Results and images will be saved to `outputs/`.

### Advanced Options

```bash
python scripts/run_experiment.py \
  --device cpu \
  --grid-size 256 \
  --timesteps 500 \
  --num-partitions 1 2 4 \
  --diffusion 0.1 \
  --reaction-rate 1.0 \
  --dt 1e-4 \
  --domain-size 1.0 \
  --seed 42 \
  --output-dir outputs \
  --save-images \
  --save-json
```

### Key Arguments

- `--device`: `cpu` or `cuda`.
- `--grid-size`: Grid size (square domain).
- `--timesteps`: Number of time steps to evolve.
- `--num-partitions`: Space-separated list of partition counts to test (default: `1 2 4`).
- `--diffusion`: Diffusion coefficient $D$.
- `--reaction-rate`: Reaction rate $k$.
- `--dt`: Time step. If omitted, computed as 80% of the stability limit.
- `--save-images`: Save heatmaps of initial, mid, and final states.
- `--save-json`: Save results as JSON for later analysis.
- `--output-dir`: Output directory for results.

### Output

The experiment produces:

1. **Console output**: Summary of conservation metrics, error metrics, and performance statistics.
2. **Heatmaps** (if `--save-images`):
   - `ref_initial.png`: Initial condition (reference).
   - `ref_final.png`: Final state (reference).
   - `p{n}_partitioned.png`, `p{n}_reference.png`, `p{n}_difference.png`: Comparisons for $n$ partitions.
3. **JSON results** (if `--save-json`):
   - `results.json`: Detailed metrics and timing data.

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run a specific test module:

```bash
pytest tests/test_solver.py -v
```

Tests cover:
- Solver initialization, stability checks, and numerical methods.
- Partition decomposition, ghost exchange, and reassembly.
- Conservation metrics and field comparisons.
- Boundary conditions and edge cases.

## Example: Single GPU Run

On a system with CUDA support:

```bash
python scripts/run_experiment.py \
  --device cuda \
  --grid-size 512 \
  --timesteps 1000 \
  --num-partitions 1 2 4 \
  --save-images \
  --save-json
```

The solver will:
1. Initialize on GPU.
2. Run reference and partitioned simulations.
3. Synchronize before/after timing.
4. Save results to `outputs/`.

## Extending the Experiment

### Future Enhancements

1. **Multiprocessing**: Use `multiprocessing.Pool` to parallelize partition updates.
2. **MPI**: Replace ghost exchange with MPI communication (`mpi4py`).
3. **Distributed PyTorch**: Use `torch.distributed` for multi-GPU or multi-node execution.
4. **Adaptive time stepping**: Implement error-based time-step control.
5. **Alternative reactions**: Generalize to other reaction terms (e.g., Gray-Scott, Turing patterns).
6. **3D extensions**: Extend to three-dimensional domains.

### Adding a Custom Reaction

Modify `reaction_diffusion/solver.py`:

```python
def reaction_term(self, u: torch.Tensor) -> torch.Tensor:
    # Replace with your reaction model
    return your_reaction_formula(u)
```

### Custom Boundary Conditions

Modify `reaction_diffusion/solver.py` in the `laplacian_5point` method to implement different boundary conditions (e.g., Dirichlet, periodic).

## Performance Considerations

### Current Implementation

This single-process implementation partitions the grid but updates all partitions sequentially in one Python process. It demonstrates **correctness and decomposition logic**, not parallel speedup.

- Single partition: Baseline performance.
- Multiple partitions: Same or slightly slower due to overhead.

### Speedup Claims

**Do not claim speedup from partition count alone.** True speedup requires:
- Multiprocessing or multithreading with proper load balancing.
- MPI or distributed execution across multiple nodes.
- GPU offloading with separate streams/devices per partition.

### Profiling

Use Python's `cProfile` or PyTorch's profiler to identify bottlenecks:

```bash
python -m cProfile -s cumtime scripts/run_experiment.py --grid-size 128 --timesteps 50
```

## References

1. **Fisher–KPP equation**: Fisher, R. A. (1937). "The Wave of Advance of Advantageous Genes." *Ann. Eugen.* 7(4), 355–369.
2. **Numerical methods for PDEs**: LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
3. **Domain decomposition**: Smith, B. F., Bjørstad, P. E., & Gropp, W. D. (1996). *Domain Decomposition: Parallel Multilevel Methods for Elliptic Partial Differential Equations*. Cambridge University Press.
4. **High-performance computing**: Gropp, W., Hoefler, T., Thakur, R., & Träff, J. L. (2014). *High-Performance Computing*. MIT Press.

## License

This project is provided as an educational demonstration. Modify and extend freely for research and learning purposes.

## Visualization Example

Initial state:

```
  u
  ^
  |
1 |     XXX
  |    XXXXX
  |   XXXXXXX
  |    XXXXX
  |     XXX
  |
0 +----+----+----+----+--> x
  0    0.3  0.5  0.7  1.0
```

After evolution (diffusion spreads, reaction grows):

```
  u
  ^
  |
1 |   XXXXXXXXX
  |  XXXXXXXXXXX
  | XXXXXXXXXXXXX
  |  XXXXXXXXXXX
  |   XXXXXXXXX
  |
0 +----+----+----+----+--> x
  0    0.3  0.5  0.7  1.0
```

## Authors

Created as a scientific-computing and HPC scaling demonstration for NERSC and beyond.

## Support & Issues

For questions or issues:
1. Check the test suite for usage examples.
2. Review the docstrings in `reaction_diffusion/*.py`.
3. Run experiments with `--save-json` to inspect detailed metrics.
