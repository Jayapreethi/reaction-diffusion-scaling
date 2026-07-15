"""
Unit tests for the core solver module.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from reaction_diffusion.solver import FisherKPPSolver


class TestFisherKPPSolver:
    """Tests for FisherKPPSolver."""
    
    @pytest.fixture
    def solver(self):
        """Create a standard test solver."""
        return FisherKPPSolver(
            grid_size=32,
            domain_size=1.0,
            diffusion=0.1,
            reaction_rate=1.0,
            dt=1e-4,
            device="cpu",
        )
    
    def test_solver_initialization(self, solver):
        """Test solver initialization."""
        assert solver.grid_size == 32
        assert solver.domain_size == 1.0
        assert solver.diffusion == 0.1
        assert solver.reaction_rate == 1.0
        assert solver.device.type == "cpu"
    
    def test_stability_check_pass(self, solver):
        """Test that stability check passes for valid dt."""
        solver.check_stability()  # Should not raise
    
    def test_stability_check_fail(self, solver):
        """Test that stability check fails for invalid dt."""
        solver.dt = 1000.0  # Very large dt
        with pytest.raises(ValueError):
            solver.check_stability()
    
    def test_gaussian_bump_initialization(self, solver):
        """Test Gaussian bump initialization."""
        u = solver.initialize_gaussian_bump()
        
        # Check shape
        assert u.shape == (solver.grid_size, solver.grid_size)
        
        # Check that it's on the correct device
        assert u.device == solver.device
        
        # Check that peak is at center
        center_idx = solver.grid_size // 2
        max_idx = torch.argmax(u.reshape(-1))
        max_i = max_idx // solver.grid_size
        max_j = max_idx % solver.grid_size
        
        # Peak should be roughly at center (within ~5 grid points)
        assert abs(max_i - center_idx) < 5
        assert abs(max_j - center_idx) < 5
    
    def test_laplacian_5point(self, solver):
        """Test 5-point Laplacian computation."""
        # Create a simple field
        u = torch.ones((solver.grid_size, solver.grid_size), device=solver.device)
        lap_u = solver.laplacian_5point(u)
        
        # Laplacian of constant should be zero everywhere
        assert torch.allclose(lap_u, torch.zeros_like(lap_u), atol=1e-6)
    
    def test_laplacian_gaussian(self, solver):
        """Test Laplacian of Gaussian bump."""
        u = solver.initialize_gaussian_bump(center_offset=0.5, amplitude=1.0, width=0.1)
        lap_u = solver.laplacian_5point(u)
        
        # Check shape
        assert lap_u.shape == u.shape
        
        # Laplacian should be negative in the center (where u is peaked)
        center_idx = solver.grid_size // 2
        assert lap_u[center_idx, center_idx] < 0
    
    def test_reaction_term(self, solver):
        """Test reaction term computation."""
        u = torch.full((solver.grid_size, solver.grid_size), 0.5, device=solver.device)
        reaction = solver.reaction_term(u)
        
        # R(0.5) = k * 0.5 * 0.5 = k * 0.25
        expected = solver.reaction_rate * 0.25
        assert torch.allclose(reaction, torch.full_like(reaction, expected), atol=1e-6)
    
    def test_reaction_at_boundaries(self, solver):
        """Test reaction term at u=0 and u=1."""
        u_zero = torch.zeros((solver.grid_size, solver.grid_size), device=solver.device)
        u_one = torch.ones((solver.grid_size, solver.grid_size), device=solver.device)
        
        reaction_zero = solver.reaction_term(u_zero)
        reaction_one = solver.reaction_term(u_one)
        
        # R(0) = 0 and R(1) = 0
        assert torch.allclose(reaction_zero, torch.zeros_like(reaction_zero), atol=1e-6)
        assert torch.allclose(reaction_one, torch.zeros_like(reaction_one), atol=1e-6)
    
    def test_step(self, solver):
        """Test one time step."""
        u = solver.initialize_gaussian_bump()
        u_new = solver.step(u)
        
        # Check shape
        assert u_new.shape == u.shape
        
        # Check that field changes
        assert not torch.allclose(u_new, u)
    
    def test_mass_computation(self, solver):
        """Test mass computation."""
        # Create uniform field
        u = torch.ones((solver.grid_size, solver.grid_size), device=solver.device)
        mass = solver.compute_mass(u)
        
        # Mass should be (grid_size²) * (dx²) = domain_size²
        expected_mass = solver.domain_size ** 2
        assert abs(mass - expected_mass) < 1e-6
    
    def test_mass_gaussian(self, solver):
        """Test mass of Gaussian bump."""
        u = solver.initialize_gaussian_bump(center_offset=0.5, amplitude=1.0, width=0.1)
        mass = solver.compute_mass(u)
        
        # Mass should be positive and reasonable
        assert mass > 0
        assert mass < solver.domain_size ** 2
    
    def test_reaction_integral(self, solver):
        """Test reaction integral computation."""
        u = torch.full((solver.grid_size, solver.grid_size), 0.5, device=solver.device)
        integral = solver.compute_reaction_integral(u)
        
        # For u=0.5, reaction = k*0.5*0.5 = k*0.25
        # Integral over domain = k*0.25 * domain_size²
        expected = solver.reaction_rate * 0.25 * (solver.domain_size ** 2)
        assert abs(integral - expected) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
