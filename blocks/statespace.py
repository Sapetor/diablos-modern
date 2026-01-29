from blocks.statespace_base import StateSpaceBaseBlock
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class StateSpaceBlock(StateSpaceBaseBlock):
    """Continuous State-Space Model block."""

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "StateSpace"

    @property
    def fn_name(self):
        return "statespace"

    @property
    def params(self):
        return {
            "A": {"default": [[0.0]], "type": "list"},
            "B": {"default": [[1.0]], "type": "list"},
            "C": {"default": [[1.0]], "type": "list"},
            "D": {"default": [[0.0]], "type": "list"},
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

    def draw_icon(self, block_rect):
        """StateSpace uses complex rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        """Execute continuous state-space block (discretized for simulation)."""
        output_only = kwargs.get('output_only', False)

        if params.get('_init_start_', True):
            params['_init_start_'] = False

            # Validate matrices
            result = self._validate_state_space_matrices(
                params['A'], params['B'], params['C'], params['D']
            )
            if isinstance(result, dict):
                return result
            A, B, C, D, n, m, p = result

            # Discretize continuous system
            dtime = params['dtime']
            try:
                Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dtime, method='zoh')
            except Exception as e:
                return {'E': True, 'error': f'Failed to discretize system: {e}'}

            params['_Ad_'] = Ad
            params['_Bd_'] = Bd
            params['_Cd_'] = Cd
            params['_Dd_'] = Dd
            params['_x_'] = self._initialize_state_vector(n, params.get('init_conds', [0.0]))
            params['_n_states_'] = n
            params['_n_inputs_'] = m
            params['_n_outputs_'] = p

        # Process input
        u, err = self._process_input(inputs, params['_n_inputs_'], output_only)
        if err:
            return err

        # Compute output
        y, err = self._compute_output(
            params['_Cd_'], params['_Dd_'], params['_x_'], u
        )
        if err:
            return err

        # Update state
        if not output_only:
            err = self._update_state(
                params['_Ad_'], params['_Bd_'], params['_x_'], u, params
            )
            if err:
                return err

        return {0: self._format_output(y), 'E': False}
