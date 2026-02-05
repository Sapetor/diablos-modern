"""
PDE Helper Functions for DiaBloS System Compiler.
Provides utilities for parsing initial conditions, handling boundary conditions,
and extracting input sources for PDE blocks.
"""

import logging
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple

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

        elif ic_lower == 'sine':
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
    ic_spec: Union[str, int, float],
    Nx: int,
    Ny: int,
    Lx: float = 1.0,
    Ly: float = 1.0,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    Parse initial conditions for 2D PDE blocks.

    Args:
        ic_spec: Initial condition specification (string or scalar)
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
    else:
        return np.full((Ny, Nx), float(ic_spec))


class BoundaryConditionHandler:
    """
    Handles boundary condition application for PDE blocks.
    Provides static methods for Dirichlet, Neumann, and Robin BCs.
    """

    @staticmethod
    def apply_dirichlet_1d(
        dc_dt: np.ndarray,
        idx: int,
        bc_val: float,
        c_val: float,
        penalty: float = 1000.0
    ) -> None:
        """
        Apply Dirichlet BC using penalty method.
        Forces boundary to match input value.

        Args:
            dc_dt: Derivative array to modify (in-place)
            idx: Index of boundary node
            bc_val: Target boundary value
            c_val: Current value at boundary
            penalty: Penalty coefficient (stiffness)
        """
        dc_dt[idx] = penalty * (bc_val - c_val)

    @staticmethod
    def apply_neumann_1d(
        dc_dt: np.ndarray,
        idx: int,
        bc_flux: float,
        c: np.ndarray,
        dx: float,
        diffusivity: float,
        source_term: float = 0.0,
        direction: str = 'left'
    ) -> None:
        """
        Apply Neumann BC (specified flux).

        Args:
            dc_dt: Derivative array to modify (in-place)
            idx: Index of boundary node
            bc_flux: Specified flux at boundary (dc/dx)
            c: Current field values
            dx: Grid spacing
            diffusivity: Diffusion coefficient (alpha for heat, D for diffusion)
            source_term: Source term at boundary (default 0)
            direction: 'left' or 'right' boundary
        """
        dx_sq = dx * dx
        N = len(c)

        if direction == 'left':
            # Second-order ghost node formula
            d2c_dx2 = (2 * c[1] - 2 * c[0] - 2 * dx * bc_flux) / dx_sq
        else:  # right
            d2c_dx2 = (2 * c[N-2] - 2 * c[N-1] + 2 * dx * bc_flux) / dx_sq

        dc_dt[idx] = diffusivity * d2c_dx2 + source_term

    @staticmethod
    def apply_robin_1d(
        dc_dt: np.ndarray,
        idx: int,
        bc_val: float,
        c_val: float,
        h: float,
        k: float,
        dx: float,
        penalty: float = 1000.0
    ) -> None:
        """
        Apply Robin BC (mixed convective).
        Robin BC: k*dT/dx = h*(T_inf - T)

        For simplicity, uses penalty method similar to Dirichlet.

        Args:
            dc_dt: Derivative array to modify (in-place)
            idx: Index of boundary node
            bc_val: Ambient/reference value (T_inf)
            c_val: Current value at boundary
            h: Convective heat transfer coefficient
            k: Thermal conductivity
            dx: Grid spacing
            penalty: Penalty coefficient
        """
        # Simplified: use penalty method to force value toward bc_val
        dc_dt[idx] = penalty * (bc_val - c_val)


def extract_input_sources(
    input_sources: List[Optional[str]],
    port_indices: List[int]
) -> Dict[int, Optional[str]]:
    """
    Extract input source keys for specified port indices.

    Replaces repeated pattern:
        src_key = input_sources[i] if len(input_sources) > i else None

    Args:
        input_sources: List of source keys (may be shorter than needed)
        port_indices: List of port indices to extract

    Returns:
        Dict mapping port index to source key (or None if unavailable)
    """
    return {
        i: input_sources[i] if i < len(input_sources) else None
        for i in port_indices
    }


def get_input_source(
    input_sources: List[Optional[str]],
    port_index: int
) -> Optional[str]:
    """
    Get a single input source key for a port index.

    Args:
        input_sources: List of source keys
        port_index: Port index to get

    Returns:
        Source key or None if unavailable
    """
    return input_sources[port_index] if port_index < len(input_sources) else None


def ensure_field_array(
    value: Union[int, float, np.ndarray, List],
    N: int,
    default: float = 0.0
) -> np.ndarray:
    """
    Ensure a value is a properly sized field array.

    Handles scalar broadcast, array size matching, and defaults.

    Args:
        value: Input value (scalar, array, or None-like)
        N: Required array size
        default: Default value for filling

    Returns:
        np.ndarray of shape (N,)
    """
    if value is None:
        return np.full(N, default)

    if isinstance(value, (int, float)):
        return np.full(N, float(value))

    arr = np.atleast_1d(value).flatten()

    if len(arr) == N:
        return arr
    elif len(arr) == 1:
        return np.full(N, arr[0])
    else:
        # Size mismatch - use first value or default
        return np.full(N, arr[0] if len(arr) > 0 else default)


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


def bc_inputs_1d() -> List[Dict[str, str]]:
    """
    Create standard 1D BC input port definitions.

    Returns:
        List of input port definitions for left and right BCs
    """
    return [
        {"name": "bc_left", "type": "float", "doc": "Left boundary value"},
        {"name": "bc_right", "type": "float", "doc": "Right boundary value"},
    ]


def bc_inputs_2d() -> List[Dict[str, str]]:
    """
    Create standard 2D BC input port definitions.

    Returns:
        List of input port definitions for all four boundary values
    """
    return [
        {"name": "bc_left", "type": "float", "doc": "Left boundary value"},
        {"name": "bc_right", "type": "float", "doc": "Right boundary value"},
        {"name": "bc_bottom", "type": "float", "doc": "Bottom boundary value"},
        {"name": "bc_top", "type": "float", "doc": "Top boundary value"},
    ]


def get_bc_values_1d(
    inputs: Dict[int, Any],
    port_offset: int = 1
) -> Tuple[float, float]:
    """
    Extract 1D boundary condition values from inputs.

    Args:
        inputs: Input dictionary from execute()
        port_offset: Starting port index for BC inputs (default 1, after source)

    Returns:
        Tuple of (bc_left, bc_right) values
    """
    bc_left = inputs.get(port_offset, 0.0)
    bc_right = inputs.get(port_offset + 1, 0.0)

    # Handle None values
    if bc_left is None:
        bc_left = 0.0
    if bc_right is None:
        bc_right = 0.0

    return float(bc_left), float(bc_right)


def get_bc_values_2d(
    inputs: Dict[int, Any],
    port_offset: int = 1
) -> Tuple[float, float, float, float]:
    """
    Extract 2D boundary condition values from inputs.

    Args:
        inputs: Input dictionary from execute()
        port_offset: Starting port index for BC inputs (default 1, after source)

    Returns:
        Tuple of (bc_left, bc_right, bc_bottom, bc_top) values
    """
    bc_left = inputs.get(port_offset, 0.0)
    bc_right = inputs.get(port_offset + 1, 0.0)
    bc_bottom = inputs.get(port_offset + 2, 0.0)
    bc_top = inputs.get(port_offset + 3, 0.0)

    # Handle None values
    if bc_left is None:
        bc_left = 0.0
    if bc_right is None:
        bc_right = 0.0
    if bc_bottom is None:
        bc_bottom = 0.0
    if bc_top is None:
        bc_top = 0.0

    return float(bc_left), float(bc_right), float(bc_bottom), float(bc_top)


def get_bc_types_1d(
    params: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Extract 1D boundary condition types from params.

    Args:
        params: Block parameters

    Returns:
        Tuple of (bc_type_left, bc_type_right)
    """
    return (
        params.get('bc_type_left', 'Dirichlet'),
        params.get('bc_type_right', 'Dirichlet')
    )


def get_bc_types_2d(
    params: Dict[str, Any]
) -> Tuple[str, str, str, str]:
    """
    Extract 2D boundary condition types from params.

    Args:
        params: Block parameters

    Returns:
        Tuple of (bc_type_left, bc_type_right, bc_type_bottom, bc_type_top)
    """
    return (
        params.get('bc_type_left', 'Dirichlet'),
        params.get('bc_type_right', 'Dirichlet'),
        params.get('bc_type_bottom', 'Dirichlet'),
        params.get('bc_type_top', 'Dirichlet')
    )
