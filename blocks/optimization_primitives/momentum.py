"""
Momentum Block

Momentum-accelerated gradient descent: v = β*v - α*∇f
Maintains velocity state internally for acceleration.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class MomentumBlock(BaseBlock):
    """
    Momentum optimizer for accelerated gradient descent.

    Update rule:
        v_new = β * v_old - α * ∇f
        x_new = x_old + v_new

    The block outputs v_new (the update to add to x).
    Velocity state is maintained internally.
    """

    @property
    def block_name(self):
        return "Momentum"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "teal"

    @property
    def doc(self):
        return (
            "Momentum-accelerated gradient descent."
            "\n\nUpdate rule: v = β*v - α*∇f"
            "\n\nParameters:"
            "\n- alpha: Learning rate (default: 0.01)"
            "\n- beta: Momentum coefficient (default: 0.9)"
            "\n\nInput: Gradient ∇f"
            "\nOutput: Update vector v (add to x for next iterate)"
            "\n\nVelocity state is maintained internally."
        )

    @property
    def params(self):
        return {
            "alpha": {
                "type": "float",
                "default": 0.01,
                "doc": "Learning rate"
            },
            "beta": {
                "type": "float",
                "default": 0.9,
                "doc": "Momentum coefficient"
            },
        }

    @property
    def inputs(self):
        return [{"name": "grad", "type": "vector"}]

    @property
    def outputs(self):
        return [{"name": "update", "type": "vector"}]

    def execute(self, time, inputs, params, **kwargs):
        try:
            grad = np.atleast_1d(inputs.get(0, [0.0])).astype(float)
            alpha = float(params.get('alpha', 0.01))
            beta = float(params.get('beta', 0.9))

            # Initialize velocity on first call
            if not params.get('_initialized_', False):
                params['_velocity_'] = np.zeros_like(grad)
                params['_initialized_'] = True

            # Handle dimension change
            v = params['_velocity_']
            if len(v) != len(grad):
                v = np.zeros_like(grad)

            # Update velocity: v = β*v - α*∇f
            v = beta * v - alpha * grad
            params['_velocity_'] = v

            return {0: v, 'E': False}

        except Exception as e:
            logger.error(f"Momentum error: {e}")
            return {0: np.array([0.0]), 'E': True, 'error': str(e)}
