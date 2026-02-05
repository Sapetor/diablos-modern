"""
Reusable parameter definition templates for DiaBloS blocks.

This module provides factory functions that generate common parameter definitions,
reducing duplication across block implementations.

Usage:
    from blocks.param_templates import init_flag_param, limit_params

    @property
    def params(self):
        return {
            "gain": {"type": "float", "default": 1.0},
            **init_flag_param(),
            **limit_params(default_min=-10, default_max=10),
        }
"""

import numpy as np
from typing import Dict, Any, List, Optional

# Type alias for parameter dictionary
ParamDict = Dict[str, Dict[str, Any]]


def init_flag_param(name: str = "_init_start_") -> ParamDict:
    """
    Create an initialization flag parameter.

    Used by blocks that need to initialize state on first execution.
    The flag is set to True initially and should be set to False after init.

    Args:
        name: Parameter name (default "_init_start_")

    Returns:
        Parameter dict with the init flag definition
    """
    return {
        name: {
            "type": "bool",
            "default": True,
            "doc": "Internal: initialization flag"
        }
    }


def init_conds_param(
    default: float = 0.0,
    param_name: str = "init_conds",
    param_type: str = "float",
    doc: str = "Initial condition value"
) -> ParamDict:
    """
    Create an initial conditions parameter.

    Args:
        default: Default initial condition value
        param_name: Parameter name (default "init_conds")
        param_type: Type string ("float", "list", etc.)
        doc: Documentation string

    Returns:
        Parameter dict with init_conds definition
    """
    return {
        param_name: {
            "type": param_type,
            "default": default,
            "doc": doc
        }
    }


def limit_params(
    default_min: float = -np.inf,
    default_max: float = np.inf,
    min_name: str = "min",
    max_name: str = "max",
    min_doc: str = "Lower limit",
    max_doc: str = "Upper limit"
) -> ParamDict:
    """
    Create min/max limit parameters.

    Used by Saturation, RateLimiter, and similar blocks.

    Args:
        default_min: Default minimum value
        default_max: Default maximum value
        min_name: Name for minimum parameter
        max_name: Name for maximum parameter
        min_doc: Documentation for min parameter
        max_doc: Documentation for max parameter

    Returns:
        Parameter dict with min/max definitions
    """
    return {
        min_name: {
            "type": "float",
            "default": default_min,
            "doc": min_doc
        },
        max_name: {
            "type": "float",
            "default": default_max,
            "doc": max_doc
        }
    }


def slew_rate_params(
    default_rising: float = np.inf,
    default_falling: float = np.inf
) -> ParamDict:
    """
    Create slew rate parameters for RateLimiter blocks.

    Args:
        default_rising: Default rising slew rate (units/sec)
        default_falling: Default falling slew rate magnitude (units/sec)

    Returns:
        Parameter dict with rising_slew and falling_slew definitions
    """
    return {
        "rising_slew": {
            "type": "float",
            "default": default_rising,
            "doc": "Max positive slope (units/sec)"
        },
        "falling_slew": {
            "type": "float",
            "default": default_falling,
            "doc": "Max negative slope magnitude (units/sec)"
        }
    }


def method_param(
    choices: List[str],
    default: str,
    param_name: str = "method",
    doc: Optional[str] = None
) -> ParamDict:
    """
    Create a method selection parameter.

    Used by blocks that support multiple algorithms (e.g., Integrator).

    Args:
        choices: List of valid method names
        default: Default method
        param_name: Parameter name (default "method")
        doc: Documentation string (auto-generated if None)

    Returns:
        Parameter dict with method definition
    """
    if doc is None:
        doc = f"Method: {', '.join(choices)}"

    return {
        param_name: {
            "type": "string",
            "default": default,
            "doc": doc,
            "choices": choices
        }
    }


def domain_params_1d(
    default_length: float = 1.0,
    default_nodes: int = 20,
    length_name: str = "L",
    nodes_name: str = "N"
) -> ParamDict:
    """
    Create 1D spatial domain parameters.

    Used by 1D PDE blocks.

    Args:
        default_length: Default domain length [m]
        default_nodes: Default number of spatial nodes
        length_name: Name for length parameter
        nodes_name: Name for nodes parameter

    Returns:
        Parameter dict with L and N definitions
    """
    return {
        length_name: {
            "type": "float",
            "default": default_length,
            "doc": "Domain length [m]"
        },
        nodes_name: {
            "type": "int",
            "default": default_nodes,
            "doc": "Number of spatial nodes"
        }
    }


def domain_params_2d(
    default_lx: float = 1.0,
    default_ly: float = 1.0,
    default_nx: int = 20,
    default_ny: int = 20
) -> ParamDict:
    """
    Create 2D spatial domain parameters.

    Used by 2D PDE blocks.

    Args:
        default_lx: Default domain length in x [m]
        default_ly: Default domain length in y [m]
        default_nx: Default nodes in x direction
        default_ny: Default nodes in y direction

    Returns:
        Parameter dict with Lx, Ly, Nx, Ny definitions
    """
    return {
        "Lx": {
            "type": "float",
            "default": default_lx,
            "doc": "Domain length in x [m]"
        },
        "Ly": {
            "type": "float",
            "default": default_ly,
            "doc": "Domain length in y [m]"
        },
        "Nx": {
            "type": "int",
            "default": default_nx,
            "doc": "Number of nodes in x direction"
        },
        "Ny": {
            "type": "int",
            "default": default_ny,
            "doc": "Number of nodes in y direction"
        }
    }


def diffusivity_param(
    default: float = 1.0,
    param_name: str = "alpha",
    doc: str = "Thermal diffusivity [m²/s]"
) -> ParamDict:
    """
    Create a diffusivity parameter.

    Args:
        default: Default diffusivity value
        param_name: Parameter name (default "alpha")
        doc: Documentation string

    Returns:
        Parameter dict with diffusivity definition
    """
    return {
        param_name: {
            "type": "float",
            "default": default,
            "doc": doc
        }
    }


def wave_speed_param(
    default: float = 1.0,
    param_name: str = "c",
    doc: str = "Wave propagation speed [m/s]"
) -> ParamDict:
    """
    Create a wave speed parameter.

    Args:
        default: Default wave speed
        param_name: Parameter name (default "c")
        doc: Documentation string

    Returns:
        Parameter dict with wave speed definition
    """
    return {
        param_name: {
            "type": "float",
            "default": default,
            "doc": doc
        }
    }


def advection_velocity_param(
    default: float = 1.0,
    param_name: str = "v",
    doc: str = "Advection velocity [m/s]"
) -> ParamDict:
    """
    Create an advection velocity parameter.

    Args:
        default: Default velocity
        param_name: Parameter name (default "v")
        doc: Documentation string

    Returns:
        Parameter dict with velocity definition
    """
    return {
        param_name: {
            "type": "float",
            "default": default,
            "doc": doc
        }
    }


def robin_bc_params(
    default_h_left: float = 10.0,
    default_h_right: float = 10.0,
    default_k: float = 1.0
) -> ParamDict:
    """
    Create Robin boundary condition parameters for 1D.

    Robin BC: -k * ∂T/∂x = h * (T - T_inf)

    Args:
        default_h_left: Default left convective coefficient
        default_h_right: Default right convective coefficient
        default_k: Default thermal conductivity

    Returns:
        Parameter dict with h_left, h_right, k_thermal definitions
    """
    return {
        "h_left": {
            "type": "float",
            "default": default_h_left,
            "doc": "Left Robin coefficient (heat transfer coeff)"
        },
        "h_right": {
            "type": "float",
            "default": default_h_right,
            "doc": "Right Robin coefficient (heat transfer coeff)"
        },
        "k_thermal": {
            "type": "float",
            "default": default_k,
            "doc": "Thermal conductivity for Robin BC [W/(m·K)]"
        }
    }


def pde_init_conds_param(
    default: str = "0.0",
    param_name: str = "init_conds",
    doc: str = "Initial conditions (scalar, list, or named pattern)"
) -> ParamDict:
    """
    Create PDE initial conditions parameter.

    Supports scalars, arrays, or named patterns like 'gaussian', 'sine', etc.

    Args:
        default: Default IC specification
        param_name: Parameter name
        doc: Documentation string

    Returns:
        Parameter dict with IC definition
    """
    return {
        param_name: {
            "type": "list",
            "default": [float(default)] if default.replace('.', '').replace('-', '').isdigit() else default,
            "doc": doc
        }
    }


def pde_2d_init_temp_param(
    default: str = "0.0",
    default_amplitude: float = 1.0
) -> ParamDict:
    """
    Create 2D PDE initial temperature parameters.

    Args:
        default: Default initial temp (number or 'sinusoidal', 'gaussian', 'hot_spot')
        default_amplitude: Default amplitude for non-uniform ICs

    Returns:
        Parameter dict with init_temp and init_amplitude definitions
    """
    return {
        "init_temp": {
            "type": "string",
            "default": default,
            "doc": "Initial temperature: number, 'sinusoidal', 'gaussian', or 'hot_spot'"
        },
        "init_amplitude": {
            "type": "float",
            "default": default_amplitude,
            "doc": "Amplitude for non-uniform initial conditions"
        }
    }


def verification_mode_param(
    default: str = "auto",
    param_name: str = "verify_mode"
) -> ParamDict:
    """
    Create a verification mode parameter for Scope blocks.

    Modes:
    - "auto": Use name-based heuristics (default, backward compatible)
    - "objective": Expect signal to decrease (optimization cost functions)
    - "comparison": Show first→last without pass/fail (PDE error signals)
    - "trajectory": Expect signal to change over time (state variables)
    - "none": Skip verification output entirely

    Args:
        default: Default verification mode
        param_name: Parameter name (default "verify_mode")

    Returns:
        Parameter dict with verify_mode definition
    """
    return {
        param_name: {
            "type": "string",
            "default": default,
            "doc": "Verification mode: auto, objective, comparison, trajectory, none",
            "choices": ["auto", "objective", "comparison", "trajectory", "none"]
        }
    }
