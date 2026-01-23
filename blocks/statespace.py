from blocks.base_block import BaseBlock
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class StateSpaceBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "StateSpace"

    @property
    def fn_name(self):
        return "statespace"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "A": {"default": [[0.0]], "type": "list"},  # State matrix (n×n)
            "B": {"default": [[1.0]], "type": "list"},  # Input matrix (n×m)
            "C": {"default": [[1.0]], "type": "list"},  # Output matrix (p×n)
            "D": {"default": [[0.0]], "type": "list"},  # Feedthrough matrix (p×m)
            "init_conds": {"default": [0.0], "type": "list"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return (
            "Continuous State-Space Model."
            "\n\ndx/dt = Ax + Bu"
            "\ny = Cx + Du"
            "\n\nParameters:"
            "\n- A, B, C, D: System matrices."
            "\n- Initial State: x(0) vector."
            "\n\nUsage:"
            "\nFor Modern Control (MIMO systems). Can model any linear system."
            "\nMatrices can be entered as nested lists: [[1, 0], [0, 1]]."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """StateSpace uses complex rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        """
        State-Space representation block.
        Continuous time discretized for simulation.
        dx/dt = Ax + Bu, y = Cx + Du
        """
        output_only = kwargs.get('output_only', False)
        
        if params.get('_init_start_', True):
            params['_init_start_'] = False

            # Get continuous-time matrices
            A = np.array(params['A'], dtype=float)
            B = np.array(params['B'], dtype=float)
            C = np.array(params['C'], dtype=float)
            D = np.array(params['D'], dtype=float)

            # Validate dimensions
            n = A.shape[0]  # Number of states
            if A.shape != (n, n):
                return {'E': True, 'error': 'A matrix must be square (n×n)'}

            if len(B.shape) == 1:
                B = B.reshape(-1, 1)
            if B.shape[0] != n:
                return {'E': True, 'error': f'B matrix must have {n} rows to match A'}

            m = B.shape[1]  # Number of inputs

            if len(C.shape) == 1:
                C = C.reshape(1, -1)
            if C.shape[1] != n:
                return {'E': True, 'error': f'C matrix must have {n} columns to match A'}

            p = C.shape[0]  # Number of outputs

            if len(D.shape) == 1:
                D = D.reshape(1, -1) if D.shape[0] > 1 else D.reshape(1, 1)
            if D.shape != (p, m):
                return {'E': True, 'error': f'D matrix must be {p}×{m} to match C and B'}

            # Discretize the system
            dtime = params['dtime']
            try:
                Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dtime, method='zoh')
            except Exception as e:
                return {'E': True, 'error': f'Failed to discretize system: {e}'}

            params['_Ad_'] = Ad
            params['_Bd_'] = Bd
            params['_Cd_'] = Cd
            params['_Dd_'] = Dd

            # Initialize state vector
            init_conds = np.atleast_1d(np.array(params.get('init_conds', [0.0]), dtype=float))
            if len(init_conds) < n:
                padded_conds = np.zeros(n)
                padded_conds[:len(init_conds)] = init_conds
                init_conds = padded_conds
            elif len(init_conds) > n:
                init_conds = init_conds[:n]

            params['_x_'] = init_conds.reshape(-1, 1)
            params['_n_states_'] = n
            params['_n_inputs_'] = m
            params['_n_outputs_'] = p

        # Get matrices and state
        Ad = params['_Ad_']
        Bd = params['_Bd_']
        Cd = params['_Cd_']
        Dd = params['_Dd_']
        x = params['_x_']

        # Get input
        if not output_only:
            u = inputs.get(0, 0.0)
            if isinstance(u, (int, float)):
                u = np.array([[u]])
            else:
                u = np.atleast_2d(u).reshape(-1, 1)

            if u.shape[0] != params['_n_inputs_']:
                return {'E': True, 'error': f"Input dimension mismatch: expected {params['_n_inputs_']}, got {u.shape[0]}"}
        else:
            u = np.zeros((params['_n_inputs_'], 1))

        # Compute output: y = Cx + Du
        try:
            y = Cd @ x + Dd @ u
        except ValueError as e:
            logger.error(f"Error in state space: {e}")
            return {'E': True, 'error': f"Output computation error: {e}"}

        # Update state: x[k+1] = Ad*x[k] + Bd*u[k]
        if not output_only:
            try:
                params['_x_'] = Ad @ x + Bd @ u
            except ValueError as e:
                logger.error(f"Error in state space state update: {e}")
                return {'E': True, 'error': f"State update error: {e}"}

        # Return output
        if y.size == 1:
            return {0: y.item(), 'E': False}
        else:
            return {0: y.flatten(), 'E': False}
