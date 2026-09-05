"""Base class for state-space based blocks with common functionality."""

from abc import abstractmethod
from blocks.base_block import BaseBlock
from blocks.input_helpers import advance_sample_time, sample_due
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StateSpaceBaseBlock(BaseBlock):
    """
    Abstract base class for state-space based control blocks.
    Provides common matrix validation, state initialization, and computation methods.
    Subclasses must implement: block_name, fn_name, params, execute
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def block_name(self):
        """Block display name - must be implemented by subclass."""
        pass

    @property
    @abstractmethod
    def fn_name(self):
        """Block function name - must be implemented by subclass."""
        pass

    @property
    @abstractmethod
    def params(self):
        """Block parameters - must be implemented by subclass."""
        pass

    @abstractmethod
    def execute(self, time, inputs, params, **kwargs):
        """Execute block - must be implemented by subclass."""
        pass

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
            return {"E": True, "error": "A matrix must be square (n×n)"}

        # Reshape B if 1D
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != n:
            return {"E": True, "error": f"B matrix must have {n} rows to match A"}

        m = B.shape[1]  # Number of inputs

        # Reshape C if 1D
        if len(C.shape) == 1:
            C = C.reshape(1, -1)
        if C.shape[1] != n:
            return {"E": True, "error": f"C matrix must have {n} columns to match A"}

        p = C.shape[0]  # Number of outputs

        # Reshape D if 1D
        if len(D.shape) == 1:
            D = D.reshape(1, -1) if D.shape[0] > 1 else D.reshape(1, 1)
        if D.shape != (p, m):
            return {"E": True, "error": f"D matrix must be {p}×{m} to match C and B"}

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
            padded[: len(init_conds)] = init_conds
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
                "E": True,
                "error": f"Input dimension mismatch: expected {n_inputs}, got {u.shape[0]}",
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
            return None, {"E": True, "error": f"Output computation error: {e}"}

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
            params_ref["_x_"] = A @ x + B @ u
            return None
        except ValueError as e:
            logger.error(f"Error in state space state update: {e}")
            return {"E": True, "error": f"State update error: {e}"}

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

    def _step(self, inputs, params, output_only=False):
        """Run one sample of a discretized A/B/C/D block.

        Every block in this family (TranFn, StateSpace, DiscreteTranFn,
        DiscreteStateSpace) performs the same three operations once the
        ``_Ad_``/``_Bd_``/``_Cd_``/``_Dd_`` matrices and the ``_x_`` state have
        been stashed in ``params`` by the block's own initialization:

            u <- input port 0, checked against ``_n_inputs_``
            y  = Cd x + Dd u          (computed from the *pre-update* state)
            x <- Ad x + Bd u          (skipped on the ``output_only`` path)

        The state update deliberately runs after the output is formed, so a
        feedthrough block reports y[k] for the state it had on entry.

        Args:
            inputs: Input dict from ``execute()``.
            params: Block params holding the discrete matrices and ``_x_``.
            output_only: True on the engine's peek path — read the output
                without consuming an input or advancing the state.

        Returns:
            tuple: ``(y_value, None)`` on success, where ``y_value`` is already
            formatted by :meth:`_format_output`; ``(None, error_dict)`` if any
            stage failed.
        """
        u, err = self._process_input(inputs, params.get("_n_inputs_", 1), output_only)
        if err is not None:
            return None, err

        x = params["_x_"]
        y, err = self._compute_output(params["_Cd_"], params["_Dd_"], x, u)
        if err is not None:
            return None, err

        y_value = self._format_output(y)

        if not output_only:
            err = self._update_state(params["_Ad_"], params["_Bd_"], x, u, params)
            if err is not None:
                return None, err

        return y_value, None

    def _sample_due(self, time, params):
        """Whether this call lands on (or past) the block's next sample instant.

        A non-positive ``sampling_time`` means "no rate": the block advances
        one sample per solver step, so every call is due.
        """
        sampling_time = params.get("sampling_time", -1.0)
        if sampling_time > 0:
            return sample_due(time, params["_next_sample_time_"])
        return True

    def _schedule_next_sample(self, time, params):
        """Advance ``_next_sample_time_`` past ``time`` in whole periods."""
        sampling_time = params.get("sampling_time", -1.0)
        if sampling_time > 0:
            advance_sample_time(params, "_next_sample_time_", time, sampling_time)
