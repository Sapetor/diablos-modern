from blocks.base_block import BaseBlock
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class TransferFunctionBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "TranFn"

    @property
    def fn_name(self):
        return "transfer_function"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "numerator": {"default": [1.0], "type": "list"},
            "denominator": {"default": [1.0, 1.0], "type": "list"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Represents a linear time-invariant system as a transfer function."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """TranFn uses B(s)/A(s) text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        """
        Transfer function block - continuous time, discretized for simulation.
        """
        output_only = kwargs.get('output_only', False)
        
        if params.get('_init_start_', True):
            params['_init_start_'] = False
            num = np.array(params['numerator'])
            den = np.array(params['denominator'])
            
            # Convert to state-space
            A, B, C, D = signal.tf2ss(num, den)
            
            # Discretize
            dtime = params['dtime']
            Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dtime)
            
            params['_Ad_'] = Ad
            params['_Bd_'] = Bd
            params['_Cd_'] = Cd
            params['_Dd_'] = Dd
            
            # State vector initialization
            num_states = Ad.shape[0]
            init_conds = np.atleast_1d(np.array(params.get('init_conds', 0.0), dtype=float))

            if len(init_conds) < num_states:
                padded_conds = np.zeros(num_states)
                padded_conds[:len(init_conds)] = init_conds
                init_conds = padded_conds
            elif len(init_conds) > num_states:
                init_conds = init_conds[:num_states]

            params['_x_'] = init_conds.reshape(-1, 1)

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

        # Compute output
        try:
            y = Cd @ x + Dd * u
        except ValueError as e:
            logger.error(f"Error in transfer function: {e}")
            return {'E': True, 'error': f"Matrix multiplication error: {e}"}

        # Update state
        if not output_only:
            try:
                params['_x_'] = Ad @ x + Bd * u
            except ValueError as e:
                logger.error(f"Error in transfer function state update: {e}")
                return {'E': True, 'error': f"State update error: {e}"}

        return {0: y.item(), 'E': False}
