"""Base class for state-space based blocks with common functionality."""
from blocks.base_block import BaseBlock
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StateSpaceBaseBlock(BaseBlock):
    """
    Base class for state-space based control blocks.
    Provides common matrix validation, state initialization, and computation methods.
    """

    def __init__(self):
        super().__init__()

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def _validate_state_space_matrices(self, A, B, C, D):
        """
        Validate state-space matrices dimensions.

        Args:
            A: State matrix (n×n)
            B: Input matrix (n×m)
            C: Output matrix (p×n)
            D: Feedthrough matrix (p×m)

        Returns:
            tuple: (A, B, C, D, n, m, p) if valid
            dict: Error dict {'E': True, 'error': msg} if invalid
        """
        # Ensure numpy arrays with float dtype
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        C = np.array(C, dtype=float)
        D = np.array(D, dtype=float)

        # Validate A is square
        n = A.shape[0]
        if A.shape != (n, n):
            return {'E': True, 'error': 'A matrix must be square (n×n)'}

        # Reshape B if 1D
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != n:
            return {'E': True, 'error': f'B matrix must have {n} rows to match A'}

        m = B.shape[1]  # Number of inputs

        # Reshape C if 1D
        if len(C.shape) == 1:
            C = C.reshape(1, -1)
        if C.shape[1] != n:
            return {'E': True, 'error': f'C matrix must have {n} columns to match A'}

        p = C.shape[0]  # Number of outputs

        # Reshape D if 1D
        if len(D.shape) == 1:
            D = D.reshape(1, -1) if D.shape[0] > 1 else D.reshape(1, 1)
        if D.shape != (p, m):
            return {'E': True, 'error': f'D matrix must be {p}×{m} to match C and B'}

        return (A, B, C, D, n, m, p)

    def _initialize_state_vector(self, n, init_conds):
        """
        Initialize state vector with proper padding/truncation.

        Args:
            n: Number of states
            init_conds: Initial conditions (list or array)

        Returns:
            np.ndarray: State vector (n×1)
        """
        init_conds = np.atleast_1d(np.array(init_conds, dtype=float))
        if len(init_conds) < n:
            padded = np.zeros(n)
            padded[:len(init_conds)] = init_conds
            init_conds = padded
        elif len(init_conds) > n:
            init_conds = init_conds[:n]
        return init_conds.reshape(-1, 1)

    def _process_input(self, inputs, n_inputs, output_only=False):
        """
        Process input signal into proper matrix form.

        Args:
            inputs: Input dict from execute()
            n_inputs: Expected number of inputs
            output_only: If True, return zeros

        Returns:
            tuple: (u, error) where u is (n_inputs×1) array or error is dict
        """
        if output_only:
            return np.zeros((n_inputs, 1)), None

        u = inputs.get(0, 0.0)
        if isinstance(u, (int, float)):
            u = np.array([[u]])
        else:
            u = np.atleast_2d(u).reshape(-1, 1)

        if u.shape[0] != n_inputs:
            return None, {
                'E': True,
                'error': f"Input dimension mismatch: expected {n_inputs}, got {u.shape[0]}"
            }

        return u, None

    def _compute_output(self, C, D, x, u):
        """
        Compute output: y = Cx + Du

        Args:
            C: Output matrix
            D: Feedthrough matrix
            x: State vector
            u: Input vector

        Returns:
            tuple: (y, error) where y is output or error is dict
        """
        try:
            y = C @ x + D @ u
            return y, None
        except ValueError as e:
            logger.error(f"Error in state space output: {e}")
            return None, {'E': True, 'error': f"Output computation error: {e}"}

    def _update_state(self, A, B, x, u, params_ref):
        """
        Update state: x[k+1] = Ax + Bu

        Args:
            A: State matrix
            B: Input matrix
            x: Current state vector
            u: Input vector
            params_ref: Reference to params dict (to update '_x_')

        Returns:
            dict or None: Error dict if failed, None if successful
        """
        try:
            params_ref['_x_'] = A @ x + B @ u
            return None
        except ValueError as e:
            logger.error(f"Error in state space state update: {e}")
            return {'E': True, 'error': f"State update error: {e}"}

    def _format_output(self, y):
        """
        Format output array to scalar or flattened array.

        Args:
            y: Output array

        Returns:
            scalar or np.ndarray
        """
        if y.size == 1:
            return y.item()
        return y.flatten()
