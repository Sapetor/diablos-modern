"""
Parameter Block - Tunable parameter for optimization

This block defines a parameter that can be optimized.
It acts as a source block during simulation, outputting its current value.
During optimization, the OptimizationEngine modifies these values.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class ParameterBlock(BaseBlock):
    """
    Tunable Parameter Block for optimization.

    During normal simulation, outputs the current parameter value.
    During optimization, the OptimizationEngine adjusts this value
    to minimize the cost function.

    Features:
    - Bounds: Constrained optimization within [lower, upper]
    - Scaling: Normalize parameter for better optimizer convergence
    - Initial value: Starting point for optimization
    """

    @property
    def block_name(self):
        return "Parameter"

    @property
    def category(self):
        return "Optimization"

    @property
    def color(self):
        return "gold"

    @property
    def doc(self):
        return (
            "Tunable Parameter for Optimization"
            "\n\nDefines a parameter that can be optimized by the Optimizer block."
            "\nActs as a constant source during simulation."
            "\n\nParameters:"
            "\n- name: Parameter name (for display and logging)"
            "\n- value: Current/initial value"
            "\n- lower: Lower bound for optimization"
            "\n- upper: Upper bound for optimization"
            "\n- scale: Scaling factor (log, linear, etc.)"
            "\n- fixed: If True, don't optimize this parameter"
            "\n\nOutputs:"
            "\n- out: Current parameter value"
        )

    @property
    def params(self):
        return {
            "name": {
                "type": "string",
                "default": "param",
                "doc": "Parameter name for identification"
            },
            "value": {
                "type": "float",
                "default": 1.0,
                "doc": "Current/initial parameter value"
            },
            "lower": {
                "type": "float",
                "default": 0.0,
                "doc": "Lower bound for optimization"
            },
            "upper": {
                "type": "float",
                "default": 10.0,
                "doc": "Upper bound for optimization"
            },
            "scale": {
                "type": "string",
                "default": "linear",
                "doc": "Scaling: 'linear', 'log', or 'normalized'"
            },
            "fixed": {
                "type": "bool",
                "default": False,
                "doc": "If True, don't optimize this parameter"
            },
        }

    @property
    def inputs(self):
        return []  # Source block, no inputs

    @property
    def outputs(self):
        return [
            {"name": "out", "type": "float", "doc": "Parameter value"},
        ]

    @property
    def requires_inputs(self):
        """Parameter is a source block."""
        return False

    def draw_icon(self, block_rect):
        """Draw parameter icon - P with adjustment slider."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw "P" letter
        path.moveTo(0.25, 0.8)
        path.lineTo(0.25, 0.2)
        path.lineTo(0.5, 0.2)
        path.cubicTo(0.7, 0.2, 0.7, 0.5, 0.5, 0.5)
        path.lineTo(0.25, 0.5)
        # Draw slider track
        path.moveTo(0.55, 0.7)
        path.lineTo(0.9, 0.7)
        # Slider knob
        path.addEllipse(0.68, 0.63, 0.14, 0.14)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Output the current parameter value."""
        value = float(params.get('value', 1.0))
        return {0: value, 'E': False}

    def get_optimization_info(self, params):
        """
        Return information for the optimizer.

        Returns:
            dict with name, value, bounds, scale, fixed
        """
        return {
            'name': params.get('name', 'param'),
            'value': float(params.get('value', 1.0)),
            'lower': float(params.get('lower', 0.0)),
            'upper': float(params.get('upper', 10.0)),
            'scale': params.get('scale', 'linear'),
            'fixed': params.get('fixed', False),
        }

    def set_value(self, params, new_value):
        """Set the parameter value (called by optimizer)."""
        params['value'] = float(new_value)

    def transform_to_optimizer(self, value, info):
        """
        Transform parameter value to optimizer space.

        Args:
            value: Physical parameter value
            info: Optimization info dict

        Returns:
            Transformed value for optimizer
        """
        scale = info.get('scale', 'linear')
        lower = info.get('lower', 0.0)
        upper = info.get('upper', 10.0)

        if scale == 'log':
            # Log transform: optimizer works in log space
            return np.log(max(value, 1e-10))
        elif scale == 'normalized':
            # Normalize to [0, 1]
            return (value - lower) / (upper - lower) if upper > lower else 0.5
        else:
            # Linear: no transform
            return value

    def transform_from_optimizer(self, opt_value, info):
        """
        Transform optimizer value back to physical space.

        Args:
            opt_value: Value from optimizer
            info: Optimization info dict

        Returns:
            Physical parameter value
        """
        scale = info.get('scale', 'linear')
        lower = info.get('lower', 0.0)
        upper = info.get('upper', 10.0)

        if scale == 'log':
            value = np.exp(opt_value)
        elif scale == 'normalized':
            value = lower + opt_value * (upper - lower)
        else:
            value = opt_value

        # Clamp to bounds
        return np.clip(value, lower, upper)
