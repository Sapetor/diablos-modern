"""
PDE Helper Functions for DiaBloS System Compiler.
Provides utilities for parsing initial conditions and building boundary-condition
parameter specs for PDE blocks.
"""

import logging
import numpy as np
from typing import Union, List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def ic_rng(seed: Union[int, float, str, None]) -> np.random.Generator:
    """RNG for the 'random' initial conditions, seeded like every other block.

    Follows the ``blocks/noise.py`` convention: ``seed == 0`` (or an unparseable
    value) means "entropy-seeded, not reproducible"; any other integer gives a
    reproducible field. Both execution paths build the initial condition through
    this helper, so a seeded 'random' IC produces the SAME field interpreted and
    compiled.
    """
    try:
        s = int(seed)
    except (TypeError, ValueError):
        s = 0
    return np.random.default_rng(s if s != 0 else None)


def companion_seed(seed: Union[int, float, str, None]) -> int:
    """Sub-seed for the SECOND random field of a two-field IC.

    The wave blocks build a displacement AND a velocity field from one ``seed``
    param. Handing both the same seed would make them identical arrays; this
    offsets the second so they are independent, while preserving the ``0 means
    entropy / not reproducible`` convention. Both execution paths call this, so
    a seeded wave IC is reproducible interpreted and compiled.
    """
    try:
        s = int(seed)
    except (TypeError, ValueError):
        s = 0
    return 0 if s == 0 else s + 1


def parse_pde_initial_condition(
    ic_spec: Union[str, int, float, List, np.ndarray],
    N: int,
    L: float = 1.0,
    pde_type: str = "heat",
    seed: Union[int, float, str, None] = 0,
) -> np.ndarray:
    """
    Parse initial conditions for PDE blocks.

    Handles: scalar, array, or string ('gaussian', 'sine', 'uniform', 'step',
    'linear', 'random').

    Args:
        ic_spec: Initial condition specification - can be:
            - scalar (int/float): Fill entire field with this value
            - array (list/ndarray): Use directly or interpolate to match N
            - string: Named initial condition pattern
        N: Number of spatial grid points
        L: Domain length (default 1.0)
        pde_type: Type of PDE ('heat', 'wave', 'advection', 'diffusion_reaction')
        seed: Seed for the 'random' pattern (0 = entropy / not reproducible)

    Returns:
        np.ndarray of shape (N,) containing initial condition values
    """
    x = np.linspace(0, L, N)

    # Handle string specifications
    if isinstance(ic_spec, str):
        ic_lower = ic_spec.lower()

        if ic_lower == "gaussian":
            # Different Gaussian shapes for different PDEs
            if pde_type == "wave":
                return np.exp(-100 * (x - L / 2) ** 2)
            elif pde_type == "advection":
                # Wider Gaussian for better numerical resolution
                return np.exp(-25 * (x - L / 4) ** 2)
            elif pde_type == "diffusion_reaction":
                return np.exp(-50 * (x - L / 2) ** 2)
            else:  # heat
                return np.exp(-50 * (x - L / 2) ** 2)

        elif ic_lower in ("sin", "sine"):
            return np.sin(np.pi * x / L)

        elif ic_lower == "uniform":
            return np.ones(N)

        elif ic_lower == "step":
            return np.where(x < L / 4, 1.0, 0.0)

        elif ic_lower == "linear":
            return 1 - x / L

        elif ic_lower == "random":
            # Uniform noise in [0, 1) -- a rough field that smooths visibly
            # under diffusion. Reproducible whenever ``seed`` is non-zero.
            return ic_rng(seed).random(N)

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
    amplitude: float = 1.0,
    seed: Union[int, float, str, None] = 0,
) -> np.ndarray:
    """
    Parse initial conditions for 2D PDE blocks.

    Named patterns: 'sinusoidal', 'gaussian', 'hot_spot', 'radial', 'linear',
    'step', 'random', 'checkerboard'. All are scaled by ``amplitude``.

    Args:
        ic_spec: Initial condition specification (string, scalar, or array)
        Nx: Number of grid points in x direction
        Ny: Number of grid points in y direction
        Lx: Domain length in x direction
        Ly: Domain length in y direction
        amplitude: Amplitude multiplier for IC pattern
        seed: Seed for the 'random' pattern (0 = entropy / not reproducible)

    Returns:
        np.ndarray of shape (Ny, Nx) containing initial condition values
    """
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)  # Shape: (Ny, Nx)

    if isinstance(ic_spec, str):
        ic_lower = ic_spec.lower()

        if ic_lower == "sinusoidal":
            # T = A * sin(pi*x/Lx) * sin(pi*y/Ly) - eigenmode of Laplacian
            return amplitude * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

        elif ic_lower == "gaussian":
            # Gaussian bump at center
            return amplitude * np.exp(-50 * ((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2))

        elif ic_lower == "hot_spot":
            # Hot spot in corner
            return amplitude * np.exp(-100 * (X**2 + Y**2))

        elif ic_lower == "radial":
            # Radial pulse from the (0,0) corner (WaveEquation2D's pattern)
            return amplitude * np.exp(-100 * (X**2 + Y**2))

        elif ic_lower == "linear":
            # Ramp along x: amplitude at x=0 falling to 0 at x=Lx
            return amplitude * (1 - X / Lx)

        elif ic_lower == "step":
            # Hot left quarter of the plate, cold elsewhere
            return amplitude * np.where(X < Lx / 4, 1.0, 0.0)

        elif ic_lower == "random":
            # Uniform noise in [0, amplitude); reproducible for non-zero seed.
            return amplitude * ic_rng(seed).random((Ny, Nx))

        elif ic_lower == "checkerboard":
            # Alternating +/- amplitude on adjacent nodes -- the highest spatial
            # frequency the grid can represent, so it decays fastest under
            # diffusion (a good stiffness / stability probe).
            i = np.arange(Nx)[None, :]
            j = np.arange(Ny)[:, None]
            return amplitude * np.where((i + j) % 2 == 0, 1.0, -1.0)

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
    include_robin: bool = True,
    options: Optional[List[str]] = None,
) -> ParamDict:
    """
    Create 1D boundary condition parameters.

    Args:
        left_default: Default BC type for left boundary
        right_default: Default BC type for right boundary
        include_robin: Include Robin BC coefficients (h_left, h_right, k_thermal)
        options: Dropdown choices for the two ``bc_type_*`` params (drives the
            QComboBox in the property editor). Defaults to Dirichlet/Neumann,
            plus Robin when ``include_robin``, plus Periodic. Pass an explicit
            list for blocks whose ``execute()`` dispatches a different set.

    Returns:
        Parameter dict with BC type definitions and optionally Robin coefficients
    """
    if options is None:
        options = ["Dirichlet", "Neumann"]
        if include_robin:
            options.append("Robin")
        options.append("Periodic")
    choices = ", ".join(options)
    params = {
        "bc_type_left": {
            "type": "string",
            "default": left_default,
            "options": list(options),
            "doc": "Left BC type: " + choices,
        },
        "bc_type_right": {
            "type": "string",
            "default": right_default,
            "options": list(options),
            "doc": "Right BC type: " + choices,
        },
    }

    if include_robin:
        params.update(
            {
                "h_left": {
                    "type": "float",
                    "default": 10.0,
                    "doc": "Left Robin coefficient (heat transfer coeff)",
                },
                "h_right": {
                    "type": "float",
                    "default": 10.0,
                    "doc": "Right Robin coefficient (heat transfer coeff)",
                },
                "k_thermal": {
                    "type": "float",
                    "default": 1.0,
                    "doc": "Thermal conductivity for Robin BC [W/(m·K)]",
                },
            }
        )

    return params


def bc_params_2d(default_type: str = "Dirichlet", include_robin: bool = False) -> ParamDict:
    """
    Create 2D boundary condition parameters.

    'Periodic' wraps a whole axis: set it on the left OR right edge to wrap x,
    on the bottom OR top edge to wrap y. The opposite edge's type is then
    ignored, and the two axes are independent (x-periodic + Dirichlet top/bottom
    is a valid channel).

    Args:
        default_type: Default BC type for all boundaries
        include_robin: Also emit the per-edge Robin coefficients
            (h_left/h_right/h_bottom/h_top, k_thermal). Only the heat family
            supports Robin; the wave family leaves this False.

    Returns:
        Parameter dict with BC type definitions for all four edges
    """
    options = ["Dirichlet", "Neumann"]
    if include_robin:
        options.append("Robin")
    options.append("Periodic")
    choices = ", ".join(options)
    params = {
        "bc_type_left": {
            "type": "string",
            "default": default_type,
            "options": list(options),
            "doc": "Left BC: " + choices,
        },
        "bc_type_right": {
            "type": "string",
            "default": default_type,
            "options": list(options),
            "doc": "Right BC: " + choices,
        },
        "bc_type_bottom": {
            "type": "string",
            "default": default_type,
            "options": list(options),
            "doc": "Bottom BC: " + choices,
        },
        "bc_type_top": {
            "type": "string",
            "default": default_type,
            "options": list(options),
            "doc": "Top BC: " + choices,
        },
    }

    if include_robin:
        params.update(robin_bc_params_2d())

    return params


def robin_bc_params_2d(default_h: float = 10.0, default_k: float = 1.0) -> ParamDict:
    """
    Per-edge Robin coefficients for the 2D heat block.

    Robin BC: ``-k dT/dn = h (T - T_inf)`` with ``n`` the OUTWARD normal, so a
    positive ``h`` always cools a plate that is hotter than ambient, on every
    edge. ``T_inf`` is the edge's ``bc_*`` value (already an input port, hence
    already time-varying); each edge gets its own ``h`` so, for example, a
    forced-convection edge and a still-air edge can coexist on one plate.

    Args:
        default_h: Default convective coefficient for every edge
        default_k: Default thermal conductivity

    Returns:
        Parameter dict with h_left, h_right, h_bottom, h_top, k_thermal
    """
    params: ParamDict = {
        edge_param: {
            "type": "float",
            "default": default_h,
            "doc": f"{edge_label} Robin coefficient h (heat transfer coeff)",
        }
        for edge_param, edge_label in (
            ("h_left", "Left"),
            ("h_right", "Right"),
            ("h_bottom", "Bottom"),
            ("h_top", "Top"),
        )
    }
    params["k_thermal"] = {
        "type": "float",
        "default": default_k,
        "doc": "Thermal conductivity for Robin BC [W/(m·K)]",
    }
    return params
