from blocks.statespace_base import StateSpaceBaseBlock
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DiscreteStateSpaceBlock(StateSpaceBaseBlock):
    """Discrete State-Space Model block with optional sampling time."""

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "DiscreteStateSpace"

    @property
    def fn_name(self):
        return "discrete_statespace"

    @property
    def params(self):
        return {
            "A": {"default": [[0.0]], "type": "list"},
            "B": {"default": [[1.0]], "type": "list"},
            "C": {"default": [[1.0]], "type": "list"},
            "D": {"default": [[0.0]], "type": "list"},
            "init_conds": {"default": [0.0], "type": "list"},
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
    def b_type(self):
        """Block type: 1=strictly proper (memory), 2=proper (direct feedthrough)."""
        return 2

    def execute(self, time, inputs, params, **kwargs):
        """Execute discrete state-space block with optional sampling time."""
        output_only = kwargs.get('output_only', False)

        if params.get('_init_start_', True):
            params['_init_start_'] = False

            # Validate matrices (already discrete, no conversion needed)
            result = self._validate_state_space_matrices(
                params['A'], params['B'], params['C'], params['D']
            )
            if isinstance(result, dict):
                return result
            A, B, C, D, n, m, p = result

            params['_Ad_'] = A
            params['_Bd_'] = B
            params['_Cd_'] = C
            params['_Dd_'] = D
            params['_x_'] = self._initialize_state_vector(n, params.get('init_conds', [0.0]))
            params['_n_states_'] = n
            params['_n_inputs_'] = m
            params['_n_outputs_'] = p
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
            return {0: params.get('_held_output_', 0.0), 'E': False}

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

        # Format and store held output
        y_val = self._format_output(y)
        params['_held_output_'] = y_val

        # Update state
        if not output_only:
            err = self._update_state(
                params['_Ad_'], params['_Bd_'], params['_x_'], u, params
            )
            if err:
                return err

            # Schedule next sample
            if sampling_time > 0:
                while params['_next_sample_time_'] <= time + 1e-9:
                    params['_next_sample_time_'] += sampling_time

        return {0: y_val, 'E': False}
