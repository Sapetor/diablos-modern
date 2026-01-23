from blocks.base_block import BaseBlock
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class DiscreteTransferFunctionBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "DiscreteTranFn"

    @property
    def fn_name(self):
        return "discrete_transfer_function"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

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
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    @property
    def b_type(self):
        """Block type: 1=strictly proper (memory), 2=proper (direct feedthrough)."""
        return 2 

    def draw_icon(self, block_rect):
        """DiscreteTranFn uses B(z)/A(z) text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        """
        Discrete Transfer function block in z-domain.
        Supports independent sampling_time (if > 0).
        """
        output_only = kwargs.get('output_only', False)
        
        if params.get('_init_start_', True):
            params['_init_start_'] = False
            num = np.array(params['numerator'])
            den = np.array(params['denominator'])
            
            # Convert to state-space
            try:
                A, B, C, D = signal.tf2ss(num, den)
            except Exception as e:
                return {'E': True, 'error': f"Error in tf2ss conversion: {e}"}
            
            params['_Ad_'] = A
            params['_Bd_'] = B
            params['_Cd_'] = C
            params['_Dd_'] = D
            
            # State vector initialization
            num_states = A.shape[0]
            init_conds = np.atleast_1d(np.array(params.get('init_conds', 0.0), dtype=float))

            if len(init_conds) < num_states:
                padded_conds = np.zeros(num_states)
                padded_conds[:len(init_conds)] = init_conds
                init_conds = padded_conds
            elif len(init_conds) > num_states:
                init_conds = init_conds[:num_states]

            params['_x_'] = init_conds.reshape(-1, 1)
            
            # Initialize timing and hold values
            params['_next_sample_time_'] = 0.0
            params['_held_output_'] = 0.0
            
            # Compute initial output (y = Cx + Du) assuming u=0 or first input?
            # We defer computation to the first sample hit below.

        # Check sampling time
        sampling_time = params.get('sampling_time', -1.0)
        
        # Determine if we should update logic
        should_update = True
        if sampling_time > 0:
            if time < params['_next_sample_time_'] - 1e-9:
                should_update = False
        
        if not should_update:
            # Return held output
            return {0: params.get('_held_output_', 0.0), 'E': False}

        # --- UPDATE LOGIC (Sample Hit) ---

        # Get discrete-time system matrices and state
        Ad = params['_Ad_']
        Bd = params['_Bd_']
        Cd = params['_Cd_']
        Dd = params['_Dd_']
        x = params['_x_']
        
        # Get input
        u = 0.0
        if not output_only:
            u = inputs.get(0, 0.0)

        # Compute output: y[k] = C*x[k] + D*u[k]
        try:
            y = Cd @ x + Dd * u
        except ValueError as e:
            logger.error(f"Error in discrete transfer function: {e}")
            return {'E': True, 'error': f"Matrix multiplication error: {e}"}

        # Store held output
        y_val = y.item()
        params['_held_output_'] = y_val

        # Update state: x[k+1] = A*x[k] + B*u[k]
        if not output_only:
            try:
                params['_x_'] = Ad @ x + Bd * u
            except ValueError as e:
                logger.error(f"Error in discrete transfer function state update: {e}")
                return {'E': True, 'error': f"State update error: {e}"}
            
            # Schedule next sample
            if sampling_time > 0:
                while params['_next_sample_time_'] <= time + 1e-9:
                    params['_next_sample_time_'] += sampling_time
        
        return {0: y_val, 'E': False}
