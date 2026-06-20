"""
PDE Helper Functions for DiaBloS System Compiler.
Provides utilities for parsing initial conditions and building boundary-condition
parameter specs for PDE blocks.
"""

import logging
import numpy as np
from typing import Union, List, Dict, Any

logger = logging.getLogger(__name__)


def parse_pde_initial_condition(
    ic_spec: Union[str, int, float, List, np.ndarray],
    N: int,
    L: float = 1.0,
    pde_type: str = 'heat'
) -> np.ndarray:
    """
    Parse initial conditions for PDE blocks.

    Handles: scalar, array, or string ('gaussian', 'sine', 'uniform', 'step', 'linear', etc.)

    Args:
        ic_spec: Initial condition specification - can be:
            - scalar (int/float): Fill entire field with this value
            - array (list/ndarray): Use directly or interpolate to match N
            - string: Named initial condition pattern
        N: Number of spatial grid points
        L: Domain length (default 1.0)
        pde_type: Type of PDE ('heat', 'wave', 'advection', 'diffusion_reaction')

    Returns:
        np.ndarray of shape (N,) containing initial condition values
    """
    x = np.linspace(0, L, N)

    # Handle string specifications
    if isinstance(ic_spec, str):
        ic_lower = ic_spec.lower()

        if ic_lower == 'gaussian':
            # Different Gaussian shapes for different PDEs
            if pde_type == 'wave':
                return np.exp(-100 * (x - L/2)**2)
            elif pde_type == 'advection':
                # Wider Gaussian for better numerical resolution
                return np.exp(-25 * (x - L/4)**2)
            elif pde_type == 'diffusion_reaction':
                return np.exp(-50 * (x - L/2)**2)
            else:  # heat
                return np.exp(-50 * (x - L/2)**2)

        elif ic_lower in ('sin', 'sine'):
            return np.sin(np.pi * x / L)

        elif ic_lower == 'uniform':
            return np.ones(N)

        elif ic_lower == 'step':
            return np.where(x < L/4, 1.0, 0.0)

        elif ic_lower == 'linear':
            return 1 - x / L

        else:
            # Try to parse as a number
            try:
                return np.full(N, float(ic_spec))
            except ValueError:
                logger.warning(f"Unknown IC specification '{ic_spec}', defaulting to zeros")
                return np.zeros(N)

    # Handle scalar values
    elif isinstance(ic_spec, (int, float)):
        return np.full(N, float(ic_spec))

    # Handle array-like values
    else:
        ic_arr = np.array(ic_spec, dtype=float).flatten()

        if len(ic_arr) == 1:
            return np.full(N, ic_arr[0])
        elif len(ic_arr) == N:
            return ic_arr
        elif len(ic_arr) < N:
            # Interpolate to match N
            x_old = np.linspace(0, 1, len(ic_arr))
            x_new = np.linspace(0, 1, N)
            return np.interp(x_new, x_old, ic_arr)
        else:
            # Subsample to match N
            indices = np.linspace(0, len(ic_arr) - 1, N, dtype=int)
            return ic_arr[indices]


def parse_pde_2d_initial_condition(
    ic_spec: Union[str, int, float, List, np.ndarray],
    Nx: int,
    Ny: int,
    Lx: float = 1.0,
    Ly: float = 1.0,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    Parse initial conditions for 2D PDE blocks.

    Args:
        ic_spec: Initial condition specification (string, scalar, or array)
        Nx: Number of grid points in x direction
        Ny: Number of grid points in y direction
        Lx: Domain length in x direction
        Ly: Domain length in y direction
        amplitude: Amplitude multiplier for IC pattern

    Returns:
        np.ndarray of shape (Ny, Nx) containing initial condition values
    """
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)  # Shape: (Ny, Nx)

    if isinstance(ic_spec, str):
        ic_lower = ic_spec.lower()

        if ic_lower == 'sinusoidal':
            # T = A * sin(pi*x/Lx) * sin(pi*y/Ly) - eigenmode of Laplacian
            return amplitude * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

        elif ic_lower == 'gaussian':
            # Gaussian bump at center
            return amplitude * np.exp(-50 * ((X - Lx/2)**2 + (Y - Ly/2)**2))

        elif ic_lower == 'hot_spot':
            # Hot spot in corner
            return amplitude * np.exp(-100 * (X**2 + Y**2))

        else:
            # Try to parse as number
            try:
                return np.full((Ny, Nx), float(ic_spec))
            except ValueError:
                return np.zeros((Ny, Nx))

    # Handle scalar values
    elif isinstance(ic_spec, (int, float)):
        return np.full((Ny, Nx), float(ic_spec))

    # Handle array-like values (mirror the 1D parser's robustness)
    else:
        ic_arr = np.array(ic_spec, dtype=float)

        # Already the right 2D shape - use directly
        if ic_arr.shape == (Ny, Nx):
            return ic_arr

        flat = ic_arr.flatten()
        if flat.size == 1:
            # Single value - broadcast to full field
            return np.full((Ny, Nx), flat[0])
        elif flat.size == Ny * Nx:
            # Right number of elements but wrong shape - reshape
            return flat.reshape((Ny, Nx))
        else:
            # Size mismatch - cannot map to grid; default to zeros and warn
            logger.warning(
                f"parse_pde_2d_initial_condition: array IC of size {flat.size} "
                f"does not match grid (Ny={Ny}, Nx={Nx}); defaulting to zeros"
            )
            return np.zeros((Ny, Nx))


# =============================================================================
# Parameter Template Factories for PDE Blocks
# =============================================================================

# Type alias for parameter dictionary
ParamDict = Dict[str, Dict[str, Any]]


def bc_params_1d(
    left_default: str = "Dirichlet",
    right_default: str = "Dirichlet",
    include_robin: bool = True
) -> ParamDict:
    """
    Create 1D boundary condition parameters.

    Args:
        left_default: Default BC type for left boundary
        right_default: Default BC type for right boundary
        include_robin: Include Robin BC coefficients (h_left, h_right, k_thermal)

    Returns:
        Parameter dict with BC type definitions and optionally Robin coefficients
    """
    params = {
        "bc_type_left": {
            "type": "string",
            "default": left_default,
            "doc": "Left BC type: Dirichlet, Neumann, or Robin"
        },
        "bc_type_right": {
            "type": "string",
            "default": right_default,
            "doc": "Right BC type: Dirichlet, Neumann, or Robin"
        }
    }

    if include_robin:
        params.update({
            "h_left": {
                "type": "float",
                "default": 10.0,
                "doc": "Left Robin coefficient (heat transfer coeff)"
            },
            "h_right": {
                "type": "float",
                "default": 10.0,
                "doc": "Right Robin coefficient (heat transfer coeff)"
            },
            "k_thermal": {
                "type": "float",
                "default": 1.0,
                "doc": "Thermal conductivity for Robin BC [W/(m·K)]"
            }
        })

    return params


def bc_params_2d(
    default_type: str = "Dirichlet"
) -> ParamDict:
    """
    Create 2D boundary condition parameters.

    Args:
        default_type: Default BC type for all boundaries

    Returns:
        Parameter dict with BC type definitions for all four edges
    """
    return {
        "bc_type_left": {
            "type": "string",
            "default": default_type,
            "doc": "Left BC: Dirichlet or Neumann"
        },
        "bc_type_right": {
            "type": "string",
            "default": default_type,
            "doc": "Right BC: Dirichlet or Neumann"
        },
        "bc_type_bottom": {
            "type": "string",
            "default": default_type,
            "doc": "Bottom BC: Dirichlet or Neumann"
        },
        "bc_type_top": {
            "type": "string",
            "default": default_type,
            "doc": "Top BC: Dirichlet or Neumann"
        }
    }
