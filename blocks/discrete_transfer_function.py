from blocks.statespace_base import StateSpaceBaseBlock
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class DiscreteTransferFunctionBlock(StateSpaceBaseBlock):
    """Discrete Transfer Function block in z-domain."""

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "DiscreteTranFn"

    @property
    def fn_name(self):
        return "discrete_transfer_function"

    @property
    def params(self):
        return {
            "numerator": {"default": [1.0, 0.0], "type": "list"},
            "denominator": {"default": [1.0, -0.5], "type": "list"},
            "sampling_time": {"default": -1.0, "type": "float"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Represents a discrete-time linear time-invariant system as a transfer function in z-domain."

    @property
    def b_type(self):
        """Block type: 1=strictly proper (memory), 2=proper (direct feedthrough)."""
        return 2

    def draw_icon(self, block_rect):
        """DiscreteTranFn uses B(z)/A(z) text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        """Execute discrete transfer function with optional sampling time."""
        output_only = kwargs.get('output_only', False)

        if params.get('_init_start_', True):
            params['_init_start_'] = False
            num = np.array(params['numerator'], dtype=float)
            den = np.array(params['denominator'], dtype=float)

            # Convert to state-space (already discrete)
            try:
                A, B, C, D = signal.tf2ss(num, den)
            except Exception as e:
                return {'E': True, 'error': f"Error in tf2ss conversion: {e}"}

            params['_Ad_'] = A
            params['_Bd_'] = B
            params['_Cd_'] = C
            params['_Dd_'] = D

            # Initialize state vector
            n = A.shape[0]
            params['_x_'] = self._initialize_state_vector(n, params.get('init_conds', [0.0]))
            params['_n_states_'] = n
            params['_n_inputs_'] = 1
            params['_n_outputs_'] = 1
            params['_next_sample_time_'] = 0.0
            params['_held_output_'] = 0.0

        # Check sampling time
        sampling_time = params.get('sampling_time', -1.0)
        should_update = True
        if sampling_time > 0:
            if time < params['_next_sample_time_'] - 1e-9:
                should_update = False

        if not should_update:
            return {0: params.get('_held_output_', 0.0), 'E': False}

        # Get input (SISO block, scalar input)
        u = 0.0
        if not output_only:
            u = inputs.get(0, 0.0)

        # Compute output: y = Cx + Du
        x = params['_x_']
        try:
            y = params['_Cd_'] @ x + params['_Dd_'] * u
        except ValueError as e:
            logger.error(f"Error in discrete transfer function: {e}")
            return {'E': True, 'error': f"Matrix multiplication error: {e}"}

        # Store held output
        y_val = y.item()
        params['_held_output_'] = y_val

        # Update state: x[k+1] = Ax + Bu
        if not output_only:
            try:
                params['_x_'] = params['_Ad_'] @ x + params['_Bd_'] * u
            except ValueError as e:
                logger.error(f"Error in discrete transfer function state update: {e}")
                return {'E': True, 'error': f"State update error: {e}"}

            # Schedule next sample
            if sampling_time > 0:
                while params['_next_sample_time_'] <= time + 1e-9:
                    params['_next_sample_time_'] += sampling_time

        return {0: y_val, 'E': False}
