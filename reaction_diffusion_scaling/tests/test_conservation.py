"""
Unit tests for conservation validation module.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from reaction_diffusion.conservation import ConservationValidator


class TestConservationValidator:
    """Tests for ConservationValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a standard validator."""
        return ConservationValidator(dx=0.1, dy=0.1, reaction_rate=1.0)
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.dx == 0.1
        assert validator.dy == 0.1
        assert validator.reaction_rate == 1.0
        assert validator.cell_area == pytest.approx(0.01)
    
    def test_compute_mass_uniform(self, validator):
        """Test mass computation for uniform field."""
        # Create 10x10 field with value 1.0
        u = torch.ones(10, 10)
        mass = validator.compute_mass(u)
        
        # Mass = sum * cell_area = 100 * 0.01 = 1.0
        assert mass == pytest.approx(1.0)
    
    def test_compute_mass_zeros(self, validator):
        """Test mass computation for zero field."""
        u = torch.zeros(10, 10)
        mass = validator.compute_mass(u)
        
        assert mass == pytest.approx(0.0)
    
    def test_compute_reaction_integral_uniform(self, validator):
        """Test reaction integral for uniform field."""
        # For u = 0.5, reaction = 1.0 * 0.5 * 0.5 = 0.25
        u = torch.ones(10, 10) * 0.5
        integral = validator.compute_reaction_integral(u)
        
        # Integral = 100 * 0.25 * 0.01 = 0.25
        expected = 0.25 * (10 * 10) * 0.01
        assert integral == pytest.approx(expected)
    
    def test_reaction_at_boundaries(self, validator):
        """Test reaction integral at boundaries."""
        u_zero = torch.zeros(10, 10)
        u_one = torch.ones(10, 10)
        
        integral_zero = validator.compute_reaction_integral(u_zero)
        integral_one = validator.compute_reaction_integral(u_one)
        
        # R(0) = 0 and R(1) = 0
        assert integral_zero == pytest.approx(0.0)
        assert integral_one == pytest.approx(0.0)
    
    def test_conservation_validation_no_change(self, validator):
        """Test conservation validation when there's no change."""
        u_initial = torch.randn(10, 10)
        u_final = u_initial.clone()
        accumulated_reaction = 0.0
        
        metrics = validator.validate_conservation(u_initial, u_final, accumulated_reaction)
        
        # Mass change should be zero
        assert metrics['m_change_actual'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['abs_residual'] == pytest.approx(0.0, abs=1e-10)
    
    def test_conservation_validation_with_reaction(self, validator):
        """Test conservation validation with reaction-driven mass change."""
        u_initial = torch.ones(10, 10) * 0.5
        # After some time, field evolves
        u_final = u_initial + 0.01
        
        # Accumulated reaction: R(0.5) * dt * domain_area
        # R(0.5) = 0.25, domain_area = 10*10*0.01 = 1.0
        accumulated_reaction = 0.25 * 1.0 * 0.01  # Approximate
        
        metrics = validator.validate_conservation(u_initial, u_final, accumulated_reaction)
        
        # Check that metrics are reasonable
        assert metrics['m_initial'] > 0
        assert metrics['m_final'] > 0
        assert 'abs_residual' in metrics
        assert 'rel_residual' in metrics
    
    def test_field_comparison_identity(self, validator):
        """Test comparison of identical fields."""
        u = torch.randn(10, 10)
        metrics = validator.compare_fields(u, u)
        
        assert metrics['max_abs_diff'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['mean_abs_diff'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['rel_l2_error'] == pytest.approx(0.0, abs=1e-10)
    
    def test_field_comparison_constant_difference(self, validator):
        """Test comparison with constant difference."""
        u1 = torch.zeros(10, 10)
        u2 = torch.ones(10, 10) * 2.0
        
        metrics = validator.compare_fields(u1, u2)
        
        assert metrics['max_abs_diff'] == pytest.approx(2.0)
        assert metrics['mean_abs_diff'] == pytest.approx(2.0)
    
    def test_field_comparison_l2_norm(self, validator):
        """Test L2 error computation."""
        u1 = torch.ones(10, 10)
        u2 = torch.ones(10, 10) * 2.0
        
        metrics = validator.compare_fields(u1, u2)
        
        # Difference = -1 everywhere
        # L2 norm of difference = sqrt(100) = 10
        # L2 norm of reference = sqrt(400) = 20
        # Relative L2 error = 10 / 20 = 0.5
        assert metrics['rel_l2_error'] == pytest.approx(0.5)
    
    def test_field_comparison_small_error(self, validator):
        """Test comparison with small relative error."""
        u_reference = torch.ones(10, 10)
        u_perturbed = u_reference * (1.0 + 1e-6)
        
        metrics = validator.compare_fields(u_perturbed, u_reference)
        
        # Relative L2 error should be approximately 1e-6
        assert metrics['rel_l2_error'] < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
