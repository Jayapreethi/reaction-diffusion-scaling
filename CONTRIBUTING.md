# Contributing to Fisher-KPP GPU Project

Thank you for your interest in contributing! This document provides guidelines for code contributions, testing, and documentation.

## Code of Conduct

- Be respectful and inclusive
- Focus on scientific accuracy
- Provide evidence (papers, benchmarks) for claims
- Help others learn

## Getting Started

1. **Fork the repository** and clone locally
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Set up development environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Testing, linting
   ```
4. **Make your changes**
5. **Run tests and validation**
6. **Submit pull request**

## Code Style

### Python Style Guide

We follow **PEP 8** with these preferences:

```python
# ✅ Good
def compute_mass(field: torch.Tensor) -> torch.Tensor:
    """
    Compute total mass (integral of field).
    
    Args:
        field: 2D tensor of shape (nx, ny)
    
    Returns:
        Scalar tensor of total mass
    """
    return field.sum() * self.dx * self.dy


# ❌ Bad
def compute_mass(field):
    return field.sum()  # What does it compute? What's the scaling?
```

### Docstring Format

Use Google-style docstrings:

```python
def step(self, u: torch.Tensor) -> torch.Tensor:
    """
    Advance solution one timestep.
    
    Args:
        u: Current solution field (nx, ny)
    
    Returns:
        Updated solution field
    
    Raises:
        ValueError: If CFL condition violated
    
    Example:
        >>> solver = FisherKPPSolver(256, device='cuda')
        >>> u = torch.randn(256, 256)
        >>> u_next = solver.step(u)
    """
```

### Type Hints

Always include type hints:

```python
# ✅ Good
def laplacian_5point(
    field: torch.Tensor,
    dx: float
) -> torch.Tensor:
    ...

# ❌ Bad
def laplacian_5point(field, dx):
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_solver.py -v

# Run with coverage report
pytest tests/ --cov=reaction_diffusion --cov-report=html

# Run specific test
pytest tests/test_solver.py::TestFisherKPPSolver::test_stability -v
```

### Writing Tests

Every new feature must include tests with **≥90% code coverage**:

```python
import pytest
import torch
from reaction_diffusion.solver import FisherKPPSolver


class TestNewFeature:
    """Test suite for new feature."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = FisherKPPSolver(
            grid_size=128,
            D=0.1,
            k=0.5,
            dt=0.01,
            device='cpu'
        )
    
    def test_feature_basic(self):
        """Test basic functionality."""
        # Arrange
        u = torch.zeros(128, 128)
        u[50:70, 50:70] = 0.5
        
        # Act
        result = self.solver.some_method(u)
        
        # Assert
        assert result.shape == u.shape
        assert torch.isfinite(result).all()
    
    def test_feature_edge_case(self):
        """Test edge cases."""
        # Test with boundary conditions, extreme values, etc.
        pass
    
    def test_feature_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            self.solver.some_method(invalid_input)
```

### Physics Validation

For numerical changes, also add physics validation tests:

```python
def test_mass_conservation(self):
    """Verify conservation of mass with new feature."""
    u = torch.ones(128, 128) * 0.1
    validator = ConservationValidator()
    
    # Run many timesteps
    for _ in range(100):
        u = self.solver.step(u)
    
    # Check conservation
    residual = validator.validate_conservation(...)
    assert residual < 1e-5, f"Conservation failed: {residual}"
```

## Benchmarking Changes

If your changes affect performance:

1. **Run baseline benchmark:**
   ```bash
   python scripts/gpu_comparison.py --grid-size 256 --timesteps 50
   ```

2. **Make your changes**

3. **Run benchmark again:**
   ```bash
   python scripts/gpu_comparison.py --grid-size 256 --timesteps 50
   ```

4. **Compare results:**
   - No regression expected for CPU time
   - GPU performance improvement acceptable
   - Physics metrics (conservation, accuracy) must not degrade

5. **Report in PR:**
   - Before/after timing
   - Performance change (%)
   - Any trade-offs

## Documentation

### Docstring Coverage

Every public function/class must have complete docstrings:

```python
class MySolver:
    """
    One-line summary.
    
    Detailed description here. Explain the algorithm,
    assumptions, and typical use cases.
    
    Attributes:
        dx: Grid spacing
        dt: Timestep
    """
    
    def __init__(self, grid_size: int, device: str = 'cpu'):
        """Initialize solver."""
```

### Adding Tests Documentation

Add new concepts to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md):

```markdown
## New Algorithm Name

Brief description of what it does and why it matters.

### Implementation Details
- Key design choice 1
- Key design choice 2

### Validation
- How we test correctness
- Physics properties preserved
```

### API Documentation

Breaking API changes require documentation updates:
- Update [docs/API.md](docs/API.md)
- Add migration guide for old code
- Update examples in README

## Performance Guidelines

### GPU Changes

If modifying GPU code:

1. **Use proper synchronization:**
   ```python
   torch.cuda.synchronize()  # Before timing GPU code
   ```

2. **Profile before optimizing:**
   ```python
   with torch.profiler.profile(...) as prof:
       # Your code
       prof.key_averages().table(sort_by='cuda_time_total')
   ```

3. **Verify numerics:**
   - Compare CPU vs GPU results (L2 error < 1e-7)
   - Check conservation residuals

### CPU Changes

For CPU optimization:

1. **Use torch.jit.script** or **torch.jit.trace** if applicable
2. **Profile with cProfile or PyTorch profiler**
3. **Verify no floating-point regressions**

## Pull Request Process

1. **Fill out PR template:**
   - What does this change?
   - Why is it needed?
   - How does it work?
   - Benchmarks (if applicable)

2. **Checks must pass:**
   - All tests pass (42/42)
   - Code coverage ≥90%
   - No lint errors (flake8)
   - Type hints complete (mypy)

3. **Request review from:**
   - Code maintainers
   - Domain scientists (for physics changes)
   - Performance experts (for optimization changes)

4. **Address feedback**

5. **Squash commits** before merge (optional but preferred)

## Specific Contribution Areas

### Bug Fixes
- Include minimal test case
- Add regression test
- Reference any related issues

### New Features
- Propose in issue first
- Include tests (≥90% coverage)
- Add documentation
- Benchmark if performance-related

### Physics Validation
- Reference scientific literature
- Show validation results
- Compare against known solutions or benchmarks

### Performance Optimization
- Before/after benchmarks
- Must not break physics
- Document trade-offs

### Documentation
- Fix typos, clarify explanations
- Add examples
- Update diagrams

## Questions?

- **Discussion:** Open a GitHub Discussion
- **Bug:** File an Issue with reproducible example
- **Feature:** Open an Issue describing the need
- **Contribution:** Email or ask in discussion first

Thank you for contributing! 🚀

---

**Last Updated:** July 2026  
**Maintainer:** [Your Name]
