"""
StateVariable Block

Holds optimization state x(k), outputs current value, accepts next value.
Each simulation step corresponds to one optimization iteration.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock
from lib.safe_eval import safe_literal

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
    def b_type(self):
        """Memory block - outputs initial state before needing input."""
        return 1

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
                "doc": "Starting value (list for vector)",
            },
            "dimension": {"type": "int", "default": 2, "doc": "Number of state variables"},
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

    def draw_icon(self, block_rect):
        """Draw a state register x fed back on itself."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        # Register box
        path.moveTo(0.26, 0.24)
        path.lineTo(0.74, 0.24)
        path.lineTo(0.74, 0.62)
        path.lineTo(0.26, 0.62)
        path.lineTo(0.26, 0.24)
        # "x"
        path.moveTo(0.38, 0.32)
        path.lineTo(0.62, 0.54)
        path.moveTo(0.62, 0.32)
        path.lineTo(0.38, 0.54)
        # Feedback loop
        path.moveTo(0.74, 0.43)
        path.lineTo(0.92, 0.43)
        path.lineTo(0.92, 0.92)
        path.lineTo(0.08, 0.92)
        path.lineTo(0.08, 0.43)
        path.lineTo(0.26, 0.43)
        path.moveTo(0.18, 0.37)
        path.lineTo(0.26, 0.43)
        path.lineTo(0.18, 0.49)
        return path

    def execute(self, time, inputs, params, **kwargs):
        try:
            # Initialize state on first call
            if params.get("_init_start_", True):
                initial = params.get("initial_value", [1.0, 1.0])
                # Do NOT silently fall back to [1.0, 1.0]: the optimization would
                # then start from a point the user never asked for and quietly
                # converge to the wrong answer. Report the bad parameter instead.
                if isinstance(initial, str):
                    try:
                        initial = safe_literal(initial)
                    except Exception as exc:
                        logger.error("StateVariable: bad initial_value %r: %s", initial, exc)
                        return {
                            "E": True,
                            "error": f"StateVariable: cannot parse initial_value {initial!r}: {exc}",
                        }
                try:
                    state = np.atleast_1d(np.array(initial, dtype=float))
                except (ValueError, TypeError) as exc:
                    logger.error("StateVariable: bad initial_value %r: %s", initial, exc)
                    return {
                        "E": True,
                        "error": f"StateVariable: initial_value {initial!r} is not numeric: {exc}",
                    }
                params["_state_"] = state
                params["_init_start_"] = False

            # Output current state
            x_current = params["_state_"].copy()

            # Accept next state for next iteration (if provided)
            x_next = inputs.get(0)
            if x_next is not None:
                params["_state_"] = np.atleast_1d(x_next).astype(float)

            return {0: x_current, "E": False}

        except Exception as e:
            logger.error(f"StateVariable error: {e}")
            dimension = int(params.get("dimension", 2))
            return {0: np.zeros(dimension), "E": True, "error": str(e)}
