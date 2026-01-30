"""
Constraint Block - Inequality/Equality constraint for optimization

This block defines constraints that must be satisfied during optimization.
Constraints can be inequality (<=, >=) or equality (==).

Used with constrained optimizers like SLSQP.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class ConstraintBlock(BaseBlock):
    """
    Constraint Block for constrained optimization.

    Defines a constraint that the optimizer must satisfy.
    The constraint value is computed from the input signal.

    Constraint types:
    - <=: g(x) <= 0 (signal <= bound)
    - >=: g(x) >= 0 (signal >= bound)
    - ==: h(x) = 0 (signal == bound)
    """

    @property
    def block_name(self):
        return "Constraint"

    @property
    def category(self):
        return "Optimization"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return (
            "Constraint for Constrained Optimization"
            "\n\nDefines a constraint that must be satisfied."
            "\n\nConstraint types:"
            "\n- '<=': signal <= bound"
            "\n- '>=': signal >= bound"
            "\n- '==': signal == bound"
            "\n\nEvaluation modes:"
            "\n- max: Constraint on maximum value over time"
            "\n- min: Constraint on minimum value over time"
            "\n- final: Constraint on final value"
            "\n- integral: Constraint on integral over time"
            "\n\nParameters:"
            "\n- type: Constraint type ('<=', '>=', '==')"
            "\n- bound: Constraint bound value"
            "\n- mode: Evaluation mode"
            "\n- tolerance: Tolerance for equality constraints"
            "\n\nInputs:"
            "\n- signal: Signal to constrain"
            "\n\nOutputs:"
            "\n- violation: Constraint violation (0 if satisfied)"
        )

    @property
    def params(self):
        return {
            "type": {
                "type": "string",
                "default": "<=",
                "doc": "Constraint type: '<=', '>=', or '=='"
            },
            "bound": {
                "type": "float",
                "default": 1.0,
                "doc": "Constraint bound value"
            },
            "mode": {
                "type": "string",
                "default": "max",
                "doc": "Evaluation: 'max', 'min', 'final', 'integral'"
            },
            "tolerance": {
                "type": "float",
                "default": 1e-6,
                "doc": "Tolerance for equality constraints"
            },
            "penalty_weight": {
                "type": "float",
                "default": 1000.0,
                "doc": "Penalty weight for constraint violation"
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
            {"name": "signal", "type": "float", "doc": "Signal to constrain"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "violation", "type": "float", "doc": "Constraint violation"},
        ]

    @property
    def requires_outputs(self):
        """Constraint is typically a terminal block."""
        return False

    def draw_icon(self, block_rect):
        """Draw constraint icon - inequality symbol with boundary."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw less-than-or-equal symbol
        path.moveTo(0.7, 0.25)
        path.lineTo(0.3, 0.4)
        path.lineTo(0.7, 0.55)
        path.moveTo(0.3, 0.65)
        path.lineTo(0.7, 0.65)
        # Draw boundary lines
        path.moveTo(0.15, 0.15)
        path.lineTo(0.15, 0.85)
        path.moveTo(0.85, 0.15)
        path.lineTo(0.85, 0.85)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Evaluate constraint and track values."""

        # Initialization
        if params.get('_init_start_', True):
            params['_max_value_'] = -np.inf
            params['_min_value_'] = np.inf
            params['_final_value_'] = 0.0
            params['_integral_'] = 0.0
            params['_init_start_'] = False

        dtime = float(params.get('dtime', 0.01))

        # Get signal
        signal = inputs.get(0, 0.0)
        if isinstance(signal, np.ndarray):
            signal = float(signal.flatten()[0])

        # Track statistics
        if signal > params.get('_max_value_', -np.inf):
            params['_max_value_'] = signal
        if signal < params.get('_min_value_', np.inf):
            params['_min_value_'] = signal
        params['_final_value_'] = signal
        params['_integral_'] = params.get('_integral_', 0.0) + signal * dtime

        # Compute current violation
        violation = self._compute_violation(signal, params)

        return {0: violation, 'E': False}

    def _compute_violation(self, signal, params):
        """Compute constraint violation for given signal value."""
        constraint_type = params.get('type', '<=')
        bound = float(params.get('bound', 1.0))
        tolerance = float(params.get('tolerance', 1e-6))

        if constraint_type == '<=':
            # g(x) = signal - bound <= 0
            violation = max(0, signal - bound)
        elif constraint_type == '>=':
            # g(x) = bound - signal <= 0 (rewritten)
            violation = max(0, bound - signal)
        elif constraint_type == '==':
            # h(x) = |signal - bound| - tolerance <= 0
            violation = max(0, abs(signal - bound) - tolerance)
        else:
            violation = 0.0

        return violation

    def get_constraint_value(self, params):
        """
        Get the constraint function value for the optimizer.

        Returns:
            (type, value) where:
            - type: 'ineq' or 'eq'
            - value: constraint function value (should be >= 0 for ineq)
        """
        mode = params.get('mode', 'max')
        constraint_type = params.get('type', '<=')
        bound = float(params.get('bound', 1.0))

        # Get the value based on mode
        if mode == 'max':
            signal = params.get('_max_value_', 0.0)
        elif mode == 'min':
            signal = params.get('_min_value_', 0.0)
        elif mode == 'final':
            signal = params.get('_final_value_', 0.0)
        elif mode == 'integral':
            signal = params.get('_integral_', 0.0)
        else:
            signal = params.get('_final_value_', 0.0)

        # Compute constraint function value
        # For scipy: inequality constraint g(x) >= 0
        if constraint_type == '<=':
            # signal <= bound -> bound - signal >= 0
            value = bound - signal
            return ('ineq', value)
        elif constraint_type == '>=':
            # signal >= bound -> signal - bound >= 0
            value = signal - bound
            return ('ineq', value)
        elif constraint_type == '==':
            # signal == bound -> signal - bound = 0
            value = signal - bound
            return ('eq', value)
        else:
            return ('ineq', 0.0)

    def get_penalty(self, params):
        """
        Get penalty value for constraint violation (for penalty methods).
        """
        mode = params.get('mode', 'max')
        penalty_weight = float(params.get('penalty_weight', 1000.0))

        if mode == 'max':
            signal = params.get('_max_value_', 0.0)
        elif mode == 'min':
            signal = params.get('_min_value_', 0.0)
        elif mode == 'final':
            signal = params.get('_final_value_', 0.0)
        elif mode == 'integral':
            signal = params.get('_integral_', 0.0)
        else:
            signal = params.get('_final_value_', 0.0)

        violation = self._compute_violation(signal, params)

        return penalty_weight * violation ** 2

    def reset(self, params):
        """Reset for a new optimization iteration."""
        params['_max_value_'] = -np.inf
        params['_min_value_'] = np.inf
        params['_final_value_'] = 0.0
        params['_integral_'] = 0.0
        params['_init_start_'] = True
