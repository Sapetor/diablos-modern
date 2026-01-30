from blocks.statespace_base import StateSpaceBaseBlock
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class TransferFunctionBlock(StateSpaceBaseBlock):
    """Continuous Transfer Function block."""

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "TranFn"

    @property
    def fn_name(self):
        return "transfer_function"

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

    def draw_icon(self, block_rect):
        """TranFn uses B(s)/A(s) text rendering - handled in DBlock switch."""
        return None

    def symbolic_execute(self, inputs, params):
        """
        Symbolic execution for equation extraction.

        In Laplace domain: Y(s) = G(s) * U(s) where G(s) = num(s) / den(s)

        Args:
            inputs: Dict of symbolic input expressions {port_idx: sympy_expr}
            params: Dict of block parameters

        Returns:
            Dict of symbolic output expressions {0: G(s) * u}
        """
        try:
            from sympy import Symbol
        except ImportError:
            return None

        s = Symbol('s')
        u = inputs.get(0, Symbol('u'))

        # Get numerator and denominator coefficients
        num = params.get('numerator', [1.0])
        den = params.get('denominator', [1.0, 1.0])

        # Ensure they are lists
        if not isinstance(num, (list, tuple)):
            num = [num]
        if not isinstance(den, (list, tuple)):
            den = [den]

        # Build polynomials in s (coefficients from highest to lowest power)
        num_poly = sum(float(coef) * s**i for i, coef in enumerate(reversed(num)))
        den_poly = sum(float(coef) * s**i for i, coef in enumerate(reversed(den)))

        # G(s) = num(s) / den(s)
        G = num_poly / den_poly

        return {0: G * u}

    def execute(self, time, inputs, params, **kwargs):
        """Execute continuous transfer function (discretized for simulation)."""
        output_only = kwargs.get('output_only', False)

        if params.get('_init_start_', True):
            params['_init_start_'] = False
            num = np.array(params['numerator'], dtype=float)
            den = np.array(params['denominator'], dtype=float)

            # Convert transfer function to state-space
            try:
                A, B, C, D = signal.tf2ss(num, den)
            except Exception as e:
                return {'E': True, 'error': f'Failed to convert TF to SS: {e}'}

            # Discretize
            dtime = params['dtime']
            try:
                Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dtime)
            except Exception as e:
                return {'E': True, 'error': f'Failed to discretize system: {e}'}

            params['_Ad_'] = Ad
            params['_Bd_'] = Bd
            params['_Cd_'] = Cd
            params['_Dd_'] = Dd

            # Initialize state vector
            n = Ad.shape[0]
            params['_x_'] = self._initialize_state_vector(n, params.get('init_conds', [0.0]))
            params['_n_states_'] = n
            params['_n_inputs_'] = 1
            params['_n_outputs_'] = 1

        # Get input (SISO block, scalar input)
        u = 0.0
        if not output_only:
            u = inputs.get(0, 0.0)

        # Compute output: y = Cx + Du
        x = params['_x_']
        try:
            y = params['_Cd_'] @ x + params['_Dd_'] * u
        except ValueError as e:
            logger.error(f"Error in transfer function: {e}")
            return {'E': True, 'error': f"Matrix multiplication error: {e}"}

        # Update state: x[k+1] = Ax + Bu
        if not output_only:
            try:
                params['_x_'] = params['_Ad_'] @ x + params['_Bd_'] * u
            except ValueError as e:
                logger.error(f"Error in transfer function state update: {e}")
                return {'E': True, 'error': f"State update error: {e}"}

        return {0: y.item(), 'E': False}
