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
    def b_type(self):
        """Feedthrough block - direct input to output."""
        return 2

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
            "alpha": {"type": "float", "default": 0.01, "doc": "Learning rate"},
            "beta": {"type": "float", "default": 0.9, "doc": "Momentum coefficient"},
        }

    @property
    def inputs(self):
        return [{"name": "grad", "type": "vector"}]

    @property
    def outputs(self):
        return [{"name": "update", "type": "vector"}]

    def draw_icon(self, block_rect):
        """Draw a cost bowl with an arrow carrying through the minimum."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        # Cost bowl
        path.moveTo(0.08, 0.10)
        path.quadTo(0.50, 1.10, 0.92, 0.10)
        # Momentum arrow through the basin
        path.moveTo(0.24, 0.66)
        path.lineTo(0.76, 0.66)
        path.moveTo(0.68, 0.58)
        path.lineTo(0.76, 0.66)
        path.lineTo(0.68, 0.74)
        return path

    def execute(self, time, inputs, params, **kwargs):
        try:
            # Output-only path: no gradient input → return last velocity without mutating state.
            if 0 not in inputs:
                held = params.get("_last_update_", np.array([0.0]))
                return {0: np.atleast_1d(held), "E": False}

            grad = np.atleast_1d(inputs.get(0, [0.0])).astype(float)
            alpha = float(params.get("alpha", 0.01))
            beta = float(params.get("beta", 0.9))

            # Initialize velocity on first call
            if params.get("_init_start_", True):
                params["_velocity_"] = np.zeros_like(grad)
                params["_init_start_"] = False

            # Handle dimension change
            v = params["_velocity_"]
            if len(v) != len(grad):
                v = np.zeros_like(grad)

            # Update velocity: v = β*v - α*∇f
            v = beta * v - alpha * grad
            params["_velocity_"] = v

            params["_last_update_"] = v
            return {0: v, "E": False}

        except Exception as e:
            logger.error(f"Momentum error: {e}")
            return {0: np.array([0.0]), "E": True, "error": str(e)}
