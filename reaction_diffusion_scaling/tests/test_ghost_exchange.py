"""
Unit tests for ghost exchange behavior and conservation properties.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from reaction_diffusion.solver import FisherKPPSolver
from reaction_diffusion.partition import StripDecomposer
from reaction_diffusion.conservation import ConservationValidator


class TestGhostExchangeBehavior:
    """Tests for ghost exchange and its correctness."""
    
    @pytest.fixture
    def setup(self):
        """Set up solver and decomposer."""
        solver = FisherKPPSolver(
            grid_size=64,
            domain_size=1.0,
            diffusion=0.1,
            reaction_rate=1.0,
            dt=1e-4,
            device="cpu",
        )
        
        decomposer = StripDecomposer(
            global_grid_size=64,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        return solver, decomposer
    
    def test_ghost_row_isolation(self, setup):
        """Test that ghost rows are not updated by local computations."""
        solver, decomposer = setup
        
        # Create partitioned fields
        u_global = solver.initialize_gaussian_bump()
        partitioned_fields = decomposer.decompose_global_field(u_global)
        
        # Store original ghosts
        p0_ghost_bottom_original = partitioned_fields[0][-1, :].clone() if decomposer.partitions[0].has_bottom_neighbor else None
        p1_ghost_top_original = partitioned_fields[1][0, :].clone() if decomposer.partitions[1].has_top_neighbor else None
        
        # Compute one step for each partition (without ghost exchange)
        for pid, partition in enumerate(decomposer.partitions):
            field = partitioned_fields[pid]
            owned_rows = partition.get_owned_rows(field)
            
            lap_u = solver.laplacian_5point(field)
            owned_rows_new = (
                owned_rows
                + solver.dt * (
                    solver.diffusion * lap_u[partition.owned_start:partition.owned_end, :]
                    + solver.reaction_term(owned_rows)
                )
            )
            field[partition.owned_start:partition.owned_end, :] = owned_rows_new
        
        # Ghost rows should not have been modified by local computation
        if p0_ghost_bottom_original is not None:
            # p0 doesn't have a bottom neighbor if decomposer has fewer partitions
            if decomposer.partitions[0].has_bottom_neighbor:
                assert torch.allclose(partitioned_fields[0][-1, :], p0_ghost_bottom_original)
    
    def test_ghost_exchange_correctness(self, setup):
        """Test that ghost exchange copies boundary rows correctly."""
        solver, decomposer = setup
        
        # Create partitioned fields with distinct values
        global_field = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64) / (64 * 64)
        partitioned_fields = decomposer.decompose_global_field(global_field)
        decomposer.exchange_ghosts(partitioned_fields)
        
        # Verify ghost rows match actual boundary rows from neighbors
        # p1's top ghost should equal p0's bottom owned row
        p0_bottom = decomposer.partitions[0].get_bottom_boundary_row(partitioned_fields[0])
        p1_top_ghost = partitioned_fields[1][0, :] if decomposer.partitions[1].has_top_neighbor else None
        
        if p1_top_ghost is not None:
            assert torch.allclose(p1_top_ghost, p0_bottom)


class TestConservationProperties:
    """Tests for mass conservation."""
    
    @pytest.fixture
    def validator(self):
        """Create a conservation validator."""
        return ConservationValidator(dx=1.0/64, dy=1.0/64, reaction_rate=1.0)
    
    def test_conservation_uniform_field(self, validator):
        """Test conservation metric for uniform field (no change)."""
        u_initial = torch.ones(64, 64)
        u_final = torch.ones(64, 64)
        accumulated_reaction_integral = 0.0
        
        metrics = validator.validate_conservation(u_initial, u_final, accumulated_reaction_integral)
        
        # Mass should not change
        assert metrics['m_change_actual'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['abs_residual'] == pytest.approx(0.0, abs=1e-6)
    
    def test_conservation_mass_increase_from_reaction(self, validator):
        """Test conservation when mass increases from reaction term."""
        u_initial = torch.ones(64, 64) * 0.5
        u_final = torch.ones(64, 64) * 0.55
        
        # Compute expected mass change
        m_initial = validator.compute_mass(u_initial)
        m_final = validator.compute_mass(u_final)
        m_change_actual = m_final - m_initial
        
        # Assume this mass change came from reaction (for testing purposes)
        accumulated_reaction = m_change_actual
        
        metrics = validator.validate_conservation(u_initial, u_final, accumulated_reaction)
        
        # With accumulated_reaction matching the actual mass change, residual should be zero
        assert metrics['abs_residual'] < 1e-10
    
    def test_conservation_metric_calculation(self, validator):
        """Test that conservation metrics are computed correctly."""
        u_initial = torch.randn(64, 64)
        u_final = torch.randn(64, 64)
        accumulated_reaction = 0.5
        
        metrics = validator.validate_conservation(u_initial, u_final, accumulated_reaction)
        
        # Check that metrics are computed
        assert 'm_initial' in metrics
        assert 'm_final' in metrics
        assert 'm_change_actual' in metrics
        assert 'm_change_expected' in metrics
        assert 'abs_residual' in metrics
        assert 'rel_residual' in metrics
        
        # Check relationship
        assert metrics['m_change_actual'] == pytest.approx(
            metrics['m_final'] - metrics['m_initial'], abs=1e-10
        )


class TestFieldComparison:
    """Tests for field comparison metrics."""
    
    @pytest.fixture
    def validator(self):
        """Create a conservation validator."""
        return ConservationValidator(dx=1.0/64, dy=1.0/64, reaction_rate=1.0)
    
    def test_identical_fields(self, validator):
        """Test comparison of identical fields."""
        u = torch.randn(64, 64)
        
        metrics = validator.compare_fields(u, u)
        
        # All differences should be zero
        assert metrics['max_abs_diff'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['mean_abs_diff'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['rel_l2_error'] == pytest.approx(0.0, abs=1e-10)
    
    def test_constant_offset(self, validator):
        """Test comparison with constant offset."""
        u1 = torch.ones(64, 64) * 1.0
        u2 = torch.ones(64, 64) * 2.0
        
        metrics = validator.compare_fields(u1, u2)
        
        # Max and mean diff should be 1.0
        assert metrics['max_abs_diff'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['mean_abs_diff'] == pytest.approx(1.0, abs=1e-6)
    
    def test_small_perturbation(self, validator):
        """Test comparison with small perturbation."""
        u_reference = torch.randn(64, 64)
        epsilon = 1e-6
        u_perturbed = u_reference + epsilon
        
        metrics = validator.compare_fields(u_perturbed, u_reference)
        
        # Max diff should be approximately epsilon (within 2% relative error)
        assert metrics['max_abs_diff'] == pytest.approx(epsilon, rel=0.02)
        assert metrics['mean_abs_diff'] == pytest.approx(epsilon, rel=0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
