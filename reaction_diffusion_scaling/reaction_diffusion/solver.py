"""
Core explicit finite-difference solver for the 2D Fisher-KPP reaction-diffusion equation.

Equation:
    ∂u/∂t = D(∂²u/∂x² + ∂²u/∂y²) + ku(1-u)

Boundary conditions: Zero-flux (Neumann) on all four boundaries.
Time integration: Explicit Euler.
Spatial discretization: Five-point Laplacian stencil.
"""

import torch
import numpy as np


class FisherKPPSolver:
    """
    Explicit finite-difference solver for 2D Fisher-KPP reaction-diffusion.
    
    Parameters:
        grid_size (int): Number of grid points along each dimension (square grid).
        domain_size (float): Physical domain size [0, domain_size] × [0, domain_size].
        diffusion (float): Diffusion coefficient D.
        reaction_rate (float): Reaction rate coefficient k.
        dt (float): Time step.
        device (str): 'cpu' or 'cuda'.
        dtype (torch.dtype): Data type for tensors.
    """
    
    def __init__(
        self,
        grid_size: int,
        domain_size: float = 1.0,
        diffusion: float = 0.1,
        reaction_rate: float = 1.0,
        dt: float = 1e-4,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.diffusion = diffusion
        self.reaction_rate = reaction_rate
        self.dt = dt
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Grid spacing
        self.dx = domain_size / grid_size
        self.dy = domain_size / grid_size
        
        # Stability condition for explicit Euler: dt ≤ dx²/(4D)
        self.stability_limit = (self.dx ** 2) / (4.0 * diffusion)
        
        # Precompute diffusion coefficient for Laplacian
        # For 5-point stencil: (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / dx²
        self.diffusion_coefficient = diffusion / (self.dx ** 2)
        
    def check_stability(self):
        """
        Verify the explicit scheme stability condition.
        Raises ValueError if dt > stability_limit.
        """
        if self.dt > self.stability_limit:
            raise ValueError(
                f"Explicit scheme is unstable. dt={self.dt:.2e} > stability_limit={self.stability_limit:.2e}. "
                f"Reduce dt or increase dx."
            )
    
    def initialize_gaussian_bump(self, center_offset: float = 0.5, amplitude: float = 1.0, width: float = 0.1) -> torch.Tensor:
        """
        Initialize the field with a Gaussian bump near the center.
        
        Parameters:
            center_offset (float): Position of the bump center as a fraction of domain_size (0 to 1).
            amplitude (float): Peak value of the Gaussian.
            width (float): Standard deviation as a fraction of domain_size.
        
        Returns:
            u (torch.Tensor): Initial field of shape (grid_size, grid_size).
        """
        # Grid coordinates
        x = torch.linspace(0, self.domain_size, self.grid_size, dtype=self.dtype, device=self.device)
        y = torch.linspace(0, self.domain_size, self.grid_size, dtype=self.dtype, device=self.device)
        
        # Bump center
        center_x = self.domain_size * center_offset
        center_y = self.domain_size * center_offset
        sigma = self.domain_size * width
        
        # Meshgrid
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Gaussian bump
        u = amplitude * torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
        
        return u
    
    def laplacian_5point(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the 2D Laplacian using the 5-point stencil with zero-flux boundary conditions.
        
        Parameters:
            u (torch.Tensor): Field of shape (grid_size, grid_size).
        
        Returns:
            lap_u (torch.Tensor): Laplacian of u.
        """
        ny, nx = u.shape
        lap_u = torch.zeros_like(u)
        
        # Interior points: standard 5-point stencil
        lap_u[1:-1, 1:-1] = (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
        ) / (self.dx ** 2)
        
        # Boundary points: zero-flux condition means ∂u/∂n = 0
        # For a Neumann boundary, we set lap_u at the boundary using the interior stencil
        # with the assumption that u_ghost = u_interior (zero flux)
        
        # Top boundary (i=0)
        lap_u[0, 1:-1] = (
            u[1, 1:-1] + u[0, 1:-1] + u[0, 2:] + u[0, :-2] - 4 * u[0, 1:-1]
        ) / (self.dx ** 2)
        
        # Bottom boundary (i=ny-1)
        lap_u[-1, 1:-1] = (
            u[-2, 1:-1] + u[-1, 1:-1] + u[-1, 2:] + u[-1, :-2] - 4 * u[-1, 1:-1]
        ) / (self.dx ** 2)
        
        # Left boundary (j=0)
        lap_u[1:-1, 0] = (
            u[2:, 0] + u[:-2, 0] + u[1:-1, 1] + u[1:-1, 0] - 4 * u[1:-1, 0]
        ) / (self.dx ** 2)
        
        # Right boundary (j=nx-1)
        lap_u[1:-1, -1] = (
            u[2:, -1] + u[:-2, -1] + u[1:-1, -2] + u[1:-1, -1] - 4 * u[1:-1, -1]
        ) / (self.dx ** 2)
        
        # Corners (handled by adjacent edge computations in many schemes)
        # For now, set corners using a simple average or diagonal stencil
        # Top-left corner
        lap_u[0, 0] = (u[1, 0] + u[0, 0] + u[0, 1] + u[0, 0] - 4 * u[0, 0]) / (self.dx ** 2)
        # Top-right corner
        lap_u[0, -1] = (u[1, -1] + u[0, -1] + u[0, -2] + u[0, -1] - 4 * u[0, -1]) / (self.dx ** 2)
        # Bottom-left corner
        lap_u[-1, 0] = (u[-2, 0] + u[-1, 0] + u[-1, 1] + u[-1, 0] - 4 * u[-1, 0]) / (self.dx ** 2)
        # Bottom-right corner
        lap_u[-1, -1] = (u[-2, -1] + u[-1, -1] + u[-1, -2] + u[-1, -1] - 4 * u[-1, -1]) / (self.dx ** 2)
        
        return lap_u
    
    def reaction_term(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the reaction term R(u) = ku(1-u).
        
        Parameters:
            u (torch.Tensor): Field of shape (grid_size, grid_size).
        
        Returns:
            R (torch.Tensor): Reaction term.
        """
        return self.reaction_rate * u * (1.0 - u)
    
    def step(self, u: torch.Tensor) -> torch.Tensor:
        """
        Advance the solution by one time step using explicit Euler.
        
        u_{n+1} = u_n + dt * (D * Δu_n + k * u_n * (1 - u_n))
        
        Parameters:
            u (torch.Tensor): Current field.
        
        Returns:
            u_new (torch.Tensor): Field after one time step.
        """
        lap_u = self.laplacian_5point(u)
        reaction = self.reaction_term(u)
        u_new = u + self.dt * (self.diffusion * lap_u + reaction)
        
        return u_new
    
    def compute_mass(self, u: torch.Tensor) -> float:
        """
        Compute the total mass (sum of field values times cell area).
        
        Parameters:
            u (torch.Tensor): Field of shape (grid_size, grid_size).
        
        Returns:
            mass (float): Total mass.
        """
        cell_area = self.dx * self.dy
        mass = (u.sum().item()) * cell_area
        return mass
    
    def compute_reaction_integral(self, u: torch.Tensor) -> float:
        """
        Compute the integral of the reaction term over the domain.
        ∫_Ω R(u) dΩ = ∫_Ω k*u*(1-u) dΩ
        
        This is used to accumulate expected mass change from reactions.
        
        Parameters:
            u (torch.Tensor): Field of shape (grid_size, grid_size).
        
        Returns:
            integral (float): Integral of reaction term.
        """
        cell_area = self.dx * self.dy
        reaction = self.reaction_term(u)
        integral = (reaction.sum().item()) * cell_area
        return integral
