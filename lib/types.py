"""
Type definitions for DiaBloS.

This module provides common type aliases used throughout the codebase
for improved code readability and type checking.
"""

from typing import Dict, List, Tuple, Union, Any, Optional, Callable, TypeVar
import numpy as np
from numpy.typing import NDArray

# Generic type variable for blocks
T = TypeVar('T')

# Block-related types
BlockParams = Dict[str, Any]
"""Dictionary of block parameters (name -> value)."""

BlockOutput = Dict[int, Union[float, int, NDArray[np.floating]]]
"""Block execution output: port index -> output value."""

BlockError = Dict[str, Union[bool, str]]
"""Block error result: {'E': True, 'error': 'message'}."""

BlockResult = Union[BlockOutput, BlockError]
"""Combined type for block execute() return value."""

# Simulation types
Timeline = NDArray[np.float64]
"""Array of time values for simulation steps."""

StateVector = NDArray[np.float64]
"""State vector for ODE integration."""

SignalValue = Union[float, int, NDArray[np.floating]]
"""A signal value: scalar or array."""

# Connection types
PortIndex = int
"""Index of an input or output port (0-based)."""

BlockName = str
"""Unique identifier for a block."""

ConnectionInfo = Dict[str, Union[str, int]]
"""Connection information: srcblock, srcport, dstblock, dstport."""

# Coordinate types
Point = Tuple[float, float]
"""2D point as (x, y) tuple."""

Rect = Tuple[float, float, float, float]
"""Rectangle as (x, y, width, height) tuple."""

# Callback types
SimulationCallback = Callable[[float, Dict[str, Any]], None]
"""Callback for simulation progress: (time, data) -> None."""

BlockExecutor = Callable[[float, Dict[int, Any], BlockParams], BlockResult]
"""Block execute function signature."""

# Matrix types for control systems
TransferFunction = Tuple[List[float], List[float]]
"""Transfer function as (numerator, denominator) coefficient lists."""

StateSpaceMatrices = Tuple[NDArray, NDArray, NDArray, NDArray]
"""State-space matrices as (A, B, C, D) tuple."""

# Field types for PDE blocks
Field1D = NDArray[np.float64]
"""1D field array for PDE solutions."""

Field2D = NDArray[np.float64]
"""2D field array for PDE solutions (shape: Ny x Nx)."""

FieldHistory = List[NDArray[np.float64]]
"""Time history of field snapshots."""

# UI types
Color = Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]
"""Color specification: name, RGB tuple, or RGBA tuple."""

# Subsystem types
NavigationPath = List[str]
"""Path through subsystem hierarchy: ['Top Level', 'Subsystem1', ...]."""

SubsystemContext = Dict[str, Any]
"""Saved subsystem context for navigation stack."""
