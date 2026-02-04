"""
Cost Function Block - Objective function accumulator

This block accumulates an error signal over time to compute
a scalar objective value for optimization.

Supported cost function types:
- ISE: Integral Squared Error = ∫(e²)dt
- IAE: Integral Absolute Error = ∫|e|dt
- ITAE: Integral Time-weighted Absolute Error = ∫t|e|dt
- terminal: Only final value matters = e(T)²
- custom: User-defined via expression
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class CostFunctionBlock(BaseBlock):
    """
    Cost Function Block for optimization.

    Accumulates error/cost over the simulation time.
    The final accumulated value is used as the objective
    for the optimizer to minimize.
    """

    @property
    def block_name(self):
        return "CostFunction"

    @property
    def category(self):
        return "Optimization"

    @property
    def color(self):
        return "red"

    @property
    def doc(self):
        return (
            "Cost Function for Optimization"
            "\n\nAccumulates a cost/error signal over time."
            "\n\nCost types:"
            "\n- ISE: ∫(e²)dt - Integral Squared Error"
            "\n- IAE: ∫|e|dt - Integral Absolute Error"
            "\n- ITAE: ∫t|e|dt - Time-weighted Absolute Error"
            "\n- terminal: e(T)² - Final value only"
            "\n- settling: Penalize settling time"
            "\n- overshoot: Penalize overshoot"
            "\n\nParameters:"
            "\n- type: Cost function type"
            "\n- target: Target/reference value"
            "\n- weight: Weight in overall objective"
            "\n- start_time: Start accumulating after this time"
            "\n\nInputs:"
            "\n- signal: Signal to evaluate (error or output)"
            "\n- reference: Reference signal (optional)"
            "\n\nOutputs:"
            "\n- cost: Current accumulated cost"
        )

    @property
    def params(self):
        return {
            "type": {
                "type": "string",
                "default": "ISE",
                "doc": "Cost type: ISE, IAE, ITAE, terminal, settling, overshoot"
            },
            "target": {
                "type": "float",
                "default": 0.0,
                "doc": "Target/reference value for error calculation"
            },
            "weight": {
                "type": "float",
                "default": 1.0,
                "doc": "Weight in overall objective function"
            },
            "start_time": {
                "type": "float",
                "default": 0.0,
                "doc": "Start accumulating after this time"
            },
            "settling_threshold": {
                "type": "float",
                "default": 0.02,
                "doc": "Settling threshold (fraction of target)"
            },
            "_init_start_": {
                "type": "bool",
                "default": True,
                "doc": "Internal: initialization flag"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "signal", "type": "float", "doc": "Signal to evaluate"},
            {"name": "reference", "type": "float", "doc": "Reference (optional)"},
        ]

    @property
    def optional_inputs(self):
        """Input 1 (reference) is optional - uses target parameter if not connected."""
        return [1]

    @property
    def outputs(self):
        return [
            {"name": "cost", "type": "float", "doc": "Accumulated cost"},
        ]

    @property
    def requires_outputs(self):
        """CostFunction is typically a terminal block."""
        return False

    def draw_icon(self, block_rect):
        """Draw cost function icon - J with target."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw "J" letter (cost function symbol)
        path.moveTo(0.55, 0.2)
        path.lineTo(0.35, 0.2)
        path.moveTo(0.45, 0.2)
        path.lineTo(0.45, 0.7)
        path.cubicTo(0.45, 0.85, 0.25, 0.85, 0.25, 0.7)
        # Draw target/bullseye
        path.addEllipse(0.6, 0.5, 0.25, 0.25)
        path.addEllipse(0.67, 0.57, 0.11, 0.11)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Accumulate cost based on input signal."""

        # Initialization
        if params.get('_init_start_', True):
            params['_accumulated_cost_'] = 0.0
            params['_prev_time_'] = 0.0
            params['_max_value_'] = 0.0
            params['_settled_'] = False
            params['_settling_time_'] = np.inf
            params['_init_start_'] = False

        dtime = float(params.get('dtime', 0.01))
        start_time = float(params.get('start_time', 0.0))
        weight = float(params.get('weight', 1.0))
        cost_type = params.get('type', 'ISE')
        target = float(params.get('target', 0.0))

        # Get signal
        signal = inputs.get(0, 0.0)
        if isinstance(signal, np.ndarray):
            signal = float(signal.flatten()[0])

        # Get reference (use target if not connected)
        reference = inputs.get(1, None)
        if reference is None:
            reference = target
        elif isinstance(reference, np.ndarray):
            reference = float(reference.flatten()[0])

        # Compute error
        error = signal - reference

        # Only accumulate after start_time
        if time < start_time:
            return {0: params.get('_accumulated_cost_', 0.0), 'E': False}

        accumulated = params.get('_accumulated_cost_', 0.0)

        if cost_type.upper() == 'ISE':
            # Integral Squared Error
            accumulated += (error ** 2) * dtime

        elif cost_type.upper() == 'IAE':
            # Integral Absolute Error
            accumulated += abs(error) * dtime

        elif cost_type.upper() == 'ITAE':
            # Integral Time-weighted Absolute Error
            accumulated += time * abs(error) * dtime

        elif cost_type.lower() == 'terminal':
            # Only final value matters (updated each step, last value used)
            accumulated = error ** 2

        elif cost_type.lower() == 'settling':
            # Settling time objective
            threshold = float(params.get('settling_threshold', 0.02))
            final_val = reference  # Assume reference is desired final value

            if abs(final_val) > 1e-10:
                rel_error = abs(error / final_val)
            else:
                rel_error = abs(error)

            if rel_error > threshold:
                params['_settling_time_'] = time
                params['_settled_'] = False
            else:
                if not params.get('_settled_', False):
                    params['_settled_'] = True

            # Cost is settling time
            accumulated = params.get('_settling_time_', np.inf)
            if accumulated == np.inf:
                accumulated = time  # Still settling

        elif cost_type.lower() == 'overshoot':
            # Penalize overshoot
            if reference > 0:
                if signal > params.get('_max_value_', 0.0):
                    params['_max_value_'] = signal
                overshoot = max(0, params['_max_value_'] - reference) / reference
            elif reference < 0:
                if signal < params.get('_max_value_', 0.0):
                    params['_max_value_'] = signal
                overshoot = max(0, reference - params['_max_value_']) / abs(reference)
            else:
                overshoot = abs(signal)

            accumulated = overshoot

        else:
            # Default to ISE
            accumulated += (error ** 2) * dtime

        params['_accumulated_cost_'] = accumulated
        params['_prev_time_'] = time

        # Apply weight
        weighted_cost = accumulated * weight

        # Debug logging
        if time < 1.1 or int(time * 10) % 10 == 0:  # Log at start and every second
            logger.info(f"CostFunction: t={time:.2f}, signal={signal:.4f}, error={error:.4f}, cost={weighted_cost:.4f}")

        return {0: weighted_cost, 'E': False}

    def get_final_cost(self, params):
        """Get the final accumulated cost (called by optimizer after simulation)."""
        weight = float(params.get('weight', 1.0))
        accumulated = params.get('_accumulated_cost_', 0.0)
        return accumulated * weight

    def reset(self, params):
        """Reset accumulated cost for a new optimization iteration."""
        params['_accumulated_cost_'] = 0.0
        params['_prev_time_'] = 0.0
        params['_max_value_'] = 0.0
        params['_settled_'] = False
        params['_settling_time_'] = np.inf
        params['_init_start_'] = True
