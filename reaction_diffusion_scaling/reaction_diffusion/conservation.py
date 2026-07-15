"""
Conservation and physics validation utilities.

Validates mass balance, compares against reference solutions, and computes error metrics.
"""

from typing import Tuple, Dict
import torch


class ConservationValidator:
    """
    Validates conservation properties and compares partitioned results against reference.
    
    Parameters:
        dx (float): Grid spacing.
        dy (float): Grid spacing.
        reaction_rate (float): Reaction rate coefficient k.
    """
    
    def __init__(self, dx: float, dy: float, reaction_rate: float):
        self.dx = dx
        self.dy = dy
        self.reaction_rate = reaction_rate
        self.cell_area = dx * dy
    
    def compute_mass(self, u: torch.Tensor) -> float:
        """
        Compute the total mass (sum of field values times cell area).
        
        Parameters:
            u (torch.Tensor): Field.
        
        Returns:
            mass (float): Total mass.
        """
        mass = (u.sum().item()) * self.cell_area
        return mass
    
    def compute_reaction_integral(self, u: torch.Tensor) -> float:
        """
        Compute the integral of the reaction term over the domain.
        ∫_Ω R(u) dΩ = ∫_Ω k*u*(1-u) dΩ
        
        Parameters:
            u (torch.Tensor): Field.
        
        Returns:
            integral (float): Integral of reaction term.
        """
        reaction = self.reaction_rate * u * (1.0 - u)
        integral = (reaction.sum().item()) * self.cell_area
        return integral
    
    def validate_conservation(
        self,
        u_initial: torch.Tensor,
        u_final: torch.Tensor,
        accumulated_reaction_integral: float,
    ) -> Dict[str, float]:
        """
        Validate mass conservation using the continuous mass balance.
        
        With no-flux boundaries:
            M(T) - M(0) = ∫_0^T ∫_Ω R(u) dΩ dt
        
        Parameters:
            u_initial (torch.Tensor): Initial field.
            u_final (torch.Tensor): Final field.
            accumulated_reaction_integral (float): Accumulated integral of R(u) over time steps.
        
        Returns:
            metrics (Dict[str, float]): Conservation metrics.
        """
        m_initial = self.compute_mass(u_initial)
        m_final = self.compute_mass(u_final)
        m_change_actual = m_final - m_initial
        m_change_expected = accumulated_reaction_integral
        
        # Absolute and relative residuals
        abs_residual = abs(m_change_actual - m_change_expected)
        rel_residual = abs_residual / (abs(m_change_expected) + 1e-16)
        
        return {
            "m_initial": m_initial,
            "m_final": m_final,
            "m_change_actual": m_change_actual,
            "m_change_expected": m_change_expected,
            "abs_residual": abs_residual,
            "rel_residual": rel_residual,
        }
    
    def compare_fields(self, u_partitioned: torch.Tensor, u_reference: torch.Tensor) -> Dict[str, float]:
        """
        Compare partitioned result against reference (single-partition) solution.
        
        Parameters:
            u_partitioned (torch.Tensor): Result from partitioned solver.
            u_reference (torch.Tensor): Reference solution.
        
        Returns:
            metrics (Dict[str, float]): Comparison metrics.
        """
        # Ensure same shape
        assert u_partitioned.shape == u_reference.shape, "Shapes must match for comparison."
        
        diff = u_partitioned - u_reference
        
        # Maximum absolute difference
        max_abs_diff = diff.abs().max().item()
        
        # Mean absolute difference
        mean_abs_diff = diff.abs().mean().item()
        
        # Relative L2 error
        l2_error_squared = (diff ** 2).sum().item()
        l2_reference_squared = (u_reference ** 2).sum().item()
        rel_l2_error = (l2_error_squared ** 0.5) / (l2_reference_squared ** 0.5 + 1e-16)
        
        return {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "rel_l2_error": rel_l2_error,
        }
