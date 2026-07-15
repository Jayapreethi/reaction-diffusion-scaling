"""
Visualization utilities for the reaction-diffusion field.

Generates heatmaps for initial, midpoint, and final states.
"""

from pathlib import Path
import torch
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def save_heatmap(
    u: torch.Tensor,
    filename: str,
    title: str = "",
    vmin: float = None,
    vmax: float = None,
    cmap: str = "viridis",
):
    """
    Save a heatmap of the 2D field.
    
    Parameters:
        u (torch.Tensor): 2D field.
        filename (str): Output file path.
        title (str): Plot title.
        vmin (float): Minimum value for colormap (auto if None).
        vmax (float): Maximum value for colormap (auto if None).
        cmap (str): Colormap name.
    """
    if not HAS_MATPLOTLIB:
        print(f"Warning: matplotlib not available. Skipping heatmap save to {filename}")
        return
    
    # Convert to numpy and detach from GPU if needed
    u_np = u.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use imshow for heatmap
    im = ax.imshow(u_np, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('u')
    
    # Save and close
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)


def save_comparison_heatmaps(
    u_partitioned: torch.Tensor,
    u_reference: torch.Tensor,
    output_dir: str,
    prefix: str = "comparison",
    vmin: float = None,
    vmax: float = None,
):
    """
    Save heatmaps comparing partitioned and reference solutions.
    
    Parameters:
        u_partitioned (torch.Tensor): Partitioned solver result.
        u_reference (torch.Tensor): Reference (single-partition) result.
        output_dir (str): Output directory.
        prefix (str): Prefix for filenames.
        vmin (float): Minimum value for colormap.
        vmax (float): Maximum value for colormap.
    """
    if not HAS_MATPLOTLIB:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use same vmin/vmax for consistency
    if vmin is None:
        vmin = float(min(u_partitioned.min(), u_reference.min()))
    if vmax is None:
        vmax = float(max(u_partitioned.max(), u_reference.max()))
    
    # Partitioned result
    save_heatmap(
        u_partitioned,
        str(output_dir / f"{prefix}_partitioned.png"),
        title="Partitioned Solution",
        vmin=vmin,
        vmax=vmax,
    )
    
    # Reference result
    save_heatmap(
        u_reference,
        str(output_dir / f"{prefix}_reference.png"),
        title="Reference Solution",
        vmin=vmin,
        vmax=vmax,
    )
    
    # Difference
    diff = (u_partitioned - u_reference).abs()
    save_heatmap(
        diff,
        str(output_dir / f"{prefix}_difference.png"),
        title="Absolute Difference",
        vmin=0.0,
        vmax=diff.max().item(),
    )
