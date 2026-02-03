"""
StateVariable Block

Holds optimization state x(k), outputs current value, accepts next value.
Each simulation step corresponds to one optimization iteration.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class StateVariableBlock(BaseBlock):
    """
    Holds the state variable for iterative optimization algorithms.

    At each time step:
    - Outputs the current state x_current
    - Accepts the next state x_next as input
    - Updates internal state for the next iteration

    This creates the feedback loop needed for iterative algorithms.
    """

    @property
    def block_name(self):
        return "StateVariable"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return (
            "Holds state x(k) for iterative optimization."
            "\n\nParameters:"
            "\n- initial_value: Starting value (list for vector, e.g., [1.0, 1.0])"
            "\n- dimension: Number of state variables"
            "\n\nInput: x_next - the next state value"
            "\nOutput: x_current - the current state value"
            "\n\nEach simulation step = one optimization iteration."
        )

    @property
    def params(self):
        return {
            "initial_value": {
                "type": "list",
                "default": [1.0, 1.0],
                "doc": "Starting value (list for vector)"
            },
            "dimension": {
                "type": "int",
                "default": 2,
                "doc": "Number of state variables"
            },
        }

    @property
    def inputs(self):
        return [{"name": "x_next", "type": "vector"}]

    @property
    def outputs(self):
        return [{"name": "x_current", "type": "vector"}]

    @property
    def requires_inputs(self):
        """State variable doesn't require input on first iteration."""
        return False

    @property
    def optional_inputs(self):
        """Input port 0 (x_next) is optional - allows execution without feedback on first step."""
        return [0]

    def execute(self, time, inputs, params, **kwargs):
        try:
            # Initialize state on first call
            if not params.get('_initialized_', False):
                initial = params.get('initial_value', [1.0, 1.0])
                if isinstance(initial, str):
                    try:
                        initial = eval(initial)
                    except Exception:
                        initial = [1.0, 1.0]
                params['_state_'] = np.array(initial, dtype=float)
                params['_initialized_'] = True

            # Output current state
            x_current = params['_state_'].copy()

            # Accept next state for next iteration (if provided)
            x_next = inputs.get(0)
            if x_next is not None:
                params['_state_'] = np.atleast_1d(x_next).astype(float)

            return {0: x_current, 'E': False}

        except Exception as e:
            logger.error(f"StateVariable error: {e}")
            dimension = int(params.get('dimension', 2))
            return {0: np.zeros(dimension), 'E': True, 'error': str(e)}
