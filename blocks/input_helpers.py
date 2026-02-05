"""
Input normalization utilities for DiaBloS blocks.

This module provides helper functions for safely extracting and normalizing
input values from the inputs dict passed to block execute() methods.

Usage:
    from blocks.input_helpers import get_scalar, get_vector, InitStateManager

    def execute(self, time, inputs, params, **kwargs):
        init_mgr = InitStateManager(params)
        if init_mgr.needs_init():
            params['mem'] = np.zeros(10)
            init_mgr.mark_initialized()

        u = get_vector(inputs, 0, expected_dim=10)
        # ... rest of execute
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Tuple


def get_scalar(
    inputs: Dict[int, Any],
    port: int,
    default: float = 0.0
) -> float:
    """
    Extract a scalar value from inputs.

    Handles None, scalars, and single-element arrays.

    Args:
        inputs: Input dictionary from execute()
        port: Port index to read from
        default: Default value if port is missing or None

    Returns:
        Scalar float value
    """
    value = inputs.get(port)

    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        return float(value)

    # Handle array-like
    arr = np.atleast_1d(value)
    if arr.size == 0:
        return float(default)
    return float(arr.flat[0])


def get_vector(
    inputs: Dict[int, Any],
    port: int,
    default: float = 0.0,
    expected_dim: Optional[int] = None
) -> np.ndarray:
    """
    Extract a vector (1D array) from inputs.

    Handles None, scalars (broadcast to expected_dim), and arrays.

    Args:
        inputs: Input dictionary from execute()
        port: Port index to read from
        default: Default value for missing/None inputs
        expected_dim: Expected array dimension (for scalar broadcast)

    Returns:
        1D numpy array
    """
    value = inputs.get(port)

    if value is None:
        if expected_dim is not None:
            return np.full(expected_dim, default)
        return np.atleast_1d(default)

    if isinstance(value, (int, float)):
        if expected_dim is not None:
            return np.full(expected_dim, float(value))
        return np.atleast_1d(float(value))

    arr = np.atleast_1d(value).flatten()

    if expected_dim is not None and len(arr) != expected_dim:
        if len(arr) == 1:
            return np.full(expected_dim, arr[0])
        # Size mismatch - interpolate or truncate
        if len(arr) < expected_dim:
            x_old = np.linspace(0, 1, len(arr))
            x_new = np.linspace(0, 1, expected_dim)
            return np.interp(x_new, x_old, arr)
        else:
            indices = np.linspace(0, len(arr) - 1, expected_dim, dtype=int)
            return arr[indices]

    return arr


def get_array_or_scalar(
    inputs: Dict[int, Any],
    port: int,
    shape: Optional[Tuple[int, ...]] = None,
    default: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Extract a value that could be scalar or array.

    If shape is provided and input is scalar, broadcasts to that shape.
    If shape is None, returns scalar for scalar input, array for array input.

    Args:
        inputs: Input dictionary from execute()
        port: Port index to read from
        shape: Target shape for broadcast (None to preserve type)
        default: Default value for missing/None inputs

    Returns:
        Scalar float or numpy array
    """
    value = inputs.get(port)

    if value is None:
        if shape is not None:
            return np.full(shape, default)
        return default

    if isinstance(value, (int, float)):
        if shape is not None:
            return np.full(shape, float(value))
        return float(value)

    arr = np.asarray(value)

    if shape is not None and arr.shape != shape:
        if arr.size == 1:
            return np.full(shape, arr.flat[0])
        # Shape mismatch - try to reshape or broadcast
        try:
            return np.broadcast_to(arr, shape).copy()
        except ValueError:
            # Can't broadcast, use first element
            return np.full(shape, arr.flat[0] if arr.size > 0 else default)

    return arr


def normalize_to_shape(
    value: Any,
    target_shape: Tuple[int, ...],
    default: float = 0.0
) -> np.ndarray:
    """
    Normalize any value to a specific array shape.

    Args:
        value: Input value (None, scalar, or array)
        target_shape: Target array shape
        default: Default value for None inputs

    Returns:
        numpy array of target_shape
    """
    if value is None:
        return np.full(target_shape, default)

    if isinstance(value, (int, float)):
        return np.full(target_shape, float(value))

    arr = np.asarray(value)

    if arr.shape == target_shape:
        return arr

    if arr.size == 1:
        return np.full(target_shape, arr.flat[0])

    # Try broadcast
    try:
        return np.broadcast_to(arr, target_shape).copy()
    except ValueError:
        pass

    # Flatten and interpolate for 1D target
    if len(target_shape) == 1:
        flat = arr.flatten()
        target_len = target_shape[0]
        if len(flat) == target_len:
            return flat
        x_old = np.linspace(0, 1, len(flat))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, flat)

    # For 2D, try reshape or use first element
    return np.full(target_shape, arr.flat[0] if arr.size > 0 else default)


class InitStateManager:
    """
    Helper class for managing block initialization state.

    Simplifies the common pattern of checking and clearing the _init_start_ flag.

    Usage:
        def execute(self, time, inputs, params, **kwargs):
            init_mgr = InitStateManager(params)
            if init_mgr.needs_init():
                # Do initialization
                params['mem'] = np.zeros(10)
                init_mgr.mark_initialized()

            # Rest of execute...
    """

    def __init__(
        self,
        params: Dict[str, Any],
        flag_name: str = "_init_start_"
    ):
        """
        Initialize the state manager.

        Args:
            params: Block parameters dict
            flag_name: Name of the init flag parameter
        """
        self._params = params
        self._flag_name = flag_name

    def needs_init(self) -> bool:
        """
        Check if initialization is needed.

        Returns:
            True if the init flag is True (or missing)
        """
        return self._params.get(self._flag_name, True)

    def mark_initialized(self) -> None:
        """
        Mark initialization as complete.

        Sets the init flag to False.
        """
        self._params[self._flag_name] = False

    def reset(self) -> None:
        """
        Reset to uninitialized state.

        Sets the init flag to True.
        """
        self._params[self._flag_name] = True


def ensure_array_size(
    arr: np.ndarray,
    target_size: int,
    default: float = 0.0
) -> np.ndarray:
    """
    Ensure a 1D array has the target size.

    Args:
        arr: Input array
        target_size: Required size
        default: Value for padding if needed

    Returns:
        Array of exactly target_size
    """
    if len(arr) == target_size:
        return arr

    if len(arr) == 1:
        return np.full(target_size, arr[0])

    if len(arr) < target_size:
        # Interpolate
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, target_size)
        return np.interp(x_new, x_old, arr)
    else:
        # Subsample
        indices = np.linspace(0, len(arr) - 1, target_size, dtype=int)
        return arr[indices]


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.

    Args:
        value: Any value
        default: Default if conversion fails

    Returns:
        Float value
    """
    if value is None:
        return default

    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return float(value.flat[0])
        return float(value)
    except (TypeError, ValueError):
        return default


def clip_to_limits(
    value: Union[float, np.ndarray],
    params: Dict[str, Any],
    min_key: str = "min",
    max_key: str = "max"
) -> Union[float, np.ndarray]:
    """
    Clip a value to min/max limits from params.

    Args:
        value: Value to clip
        params: Parameters containing min/max
        min_key: Key for minimum value
        max_key: Key for maximum value

    Returns:
        Clipped value
    """
    lower = params.get(min_key, -np.inf)
    upper = params.get(max_key, np.inf)
    return np.clip(value, lower, upper)
