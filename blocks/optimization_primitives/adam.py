"""
Adam Block

Adam optimizer with adaptive learning rates.
Combines momentum with RMSprop for robust optimization.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class AdamBlock(BaseBlock):
    """
    Adam optimizer for adaptive learning rate optimization.

    Update rules:
        m = β₁*m + (1-β₁)*∇f           (first moment)
        v = β₂*v + (1-β₂)*∇f²          (second moment)
        m̂ = m / (1 - β₁ᵗ)              (bias correction)
        v̂ = v / (1 - β₂ᵗ)              (bias correction)
        update = -α * m̂ / (√v̂ + ε)

    The block outputs the update vector to add to x.
    Moment states are maintained internally.
    """

    @property
    def block_name(self):
        return "Adam"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "pink"

    @property
    def b_type(self):
        """Feedthrough block - direct input to output."""
        return 2

    @property
    def doc(self):
        return (
            "Adam optimizer with adaptive learning rates."
            "\n\nParameters:"
            "\n- alpha: Learning rate (default: 0.001)"
            "\n- beta1: First moment decay (default: 0.9)"
            "\n- beta2: Second moment decay (default: 0.999)"
            "\n- epsilon: Numerical stability (default: 1e-8)"
            "\n\nInput: Gradient ∇f"
            "\nOutput: Update vector (add to x for next iterate)"
            "\n\nMoment states are maintained internally."
        )

    @property
    def params(self):
        return {
            "alpha": {
                "type": "float",
                "default": 0.001,
                "doc": "Learning rate"
            },
            "beta1": {
                "type": "float",
                "default": 0.9,
                "doc": "First moment decay"
            },
            "beta2": {
                "type": "float",
                "default": 0.999,
                "doc": "Second moment decay"
            },
            "epsilon": {
                "type": "float",
                "default": 1e-8,
                "doc": "Numerical stability constant"
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
            alpha = float(params.get('alpha', 0.001))
            beta1 = float(params.get('beta1', 0.9))
            beta2 = float(params.get('beta2', 0.999))
            epsilon = float(params.get('epsilon', 1e-8))

            # Initialize moments on first call
            if not params.get('_initialized_', False):
                params['_m_'] = np.zeros_like(grad)  # First moment
                params['_v_'] = np.zeros_like(grad)  # Second moment
                params['_t_'] = 0  # Time step
                params['_initialized_'] = True

            # Handle dimension change
            m = params['_m_']
            v = params['_v_']
            if len(m) != len(grad):
                m = np.zeros_like(grad)
                v = np.zeros_like(grad)
                params['_t_'] = 0

            # Increment time step
            params['_t_'] += 1
            t = params['_t_']

            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            params['_m_'] = m

            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            params['_v_'] = v

            # Compute bias-corrected estimates
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Compute update
            update = -alpha * m_hat / (np.sqrt(v_hat) + epsilon)

            return {0: update, 'E': False}

        except Exception as e:
            logger.error(f"Adam error: {e}")
            return {0: np.array([0.0]), 'E': True, 'error': str(e)}
