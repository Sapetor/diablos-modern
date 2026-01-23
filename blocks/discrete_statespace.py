from blocks.base_block import BaseBlock
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DiscreteStateSpaceBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "DiscreteStateSpace"

    @property
    def fn_name(self):
        return "discrete_statespace"

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
            "init_conds": {"default": [0.0], "type": "list"},  # Initial state vector (n×1)
            "sampling_time": {"default": -1.0, "type": "float"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return (
            "Discrete State-Space Model."
            "\n\nx[k+1] = Ax[k] + Bu[k]"
            "\ny[k] = Cx[k] + Du[k]"
            "\n\nParameters:"
            "\n- A, B, C, D: Discrete system matrices."
            "\n- Sampling Time: Execution rate."
            "\n\nUsage:"
            "\nDigital Modern Control (MIMO)."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    @property
    def b_type(self):
        """Block type: 1=strictly proper (memory), 2=proper (direct feedthrough)."""
        return 2

    def execute(self, time, inputs, params, **kwargs):
        """
        Discrete State-Space representation block.
        x[k+1] = Ax[k] + Bu[k], y[k] = Cx[k] + Du[k]
        Supports independent sampling_time.
        """
        output_only = kwargs.get('output_only', False)
        
        if params.get('_init_start_', True):
            params['_init_start_'] = False

            # Get matrices
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

            params['_Ad_'] = A
            params['_Bd_'] = B
            params['_Cd_'] = C
            params['_Dd_'] = D

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
            
            # Initialize timing and hold values
            params['_next_sample_time_'] = 0.0
            params['_held_output_'] = 0.0 if p == 1 else np.zeros(p)

        # Check sampling time
        sampling_time = params.get('sampling_time', -1.0)
        
        should_update = True
        if sampling_time > 0:
            if time < params['_next_sample_time_'] - 1e-9:
                should_update = False
        
        if not should_update:
            # Return held output
            y_held = params.get('_held_output_', 0.0)
            if isinstance(y_held, np.ndarray):
                 return {0: y_held, 'E': False}
            return {0: y_held, 'E': False}

        # --- UPDATE LOGIC ---

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
            logger.error(f"Error in discrete state space: {e}")
            return {'E': True, 'error': f"Output computation error: {e}"}
            
        # Store held output
        if y.size == 1:
            y_val = y.item()
        else:
            y_val = y.flatten()
        params['_held_output_'] = y_val

        # Update state: x[k+1] = Ax[k] + Bu[k]
        if not output_only:
            try:
                params['_x_'] = Ad @ x + Bd @ u
            except ValueError as e:
                logger.error(f"Error in discrete state space state update: {e}")
                return {'E': True, 'error': f"State update error: {e}"}
            
            # Schedule next sample
            if sampling_time > 0:
                while params['_next_sample_time_'] <= time + 1e-9:
                    params['_next_sample_time_'] += sampling_time

        # Return output
        return {0: y_val, 'E': False}
