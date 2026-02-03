"""
VectorGain Block

Scales a vector by a scalar gain: y = α * x
Used for learning rate scaling in optimization algorithms.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class VectorGainBlock(BaseBlock):
    """
    Scales a vector by a scalar gain.

    y = gain * x (element-wise scaling)

    Commonly used for learning rate scaling in gradient descent:
    update = -alpha * gradient
    """

    @property
    def block_name(self):
        return "VectorGain"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "yellow"

    @property
    def doc(self):
        return (
            "Scales a vector by a scalar gain: y = α * x"
            "\n\nParameters:"
            "\n- gain: Scalar multiplier (e.g., learning rate)"
            "\n\nInput: Vector x"
            "\nOutput: Vector y = gain * x"
            "\n\nUse negative gain for gradient descent: y = -α * ∇f"
        )

    @property
    def params(self):
        return {
            "gain": {
                "type": "float",
                "default": 1.0,
                "doc": "Scalar multiplier"
            },
        }

    @property
    def inputs(self):
        return [{"name": "x", "type": "vector"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "vector"}]

    def execute(self, time, inputs, params, **kwargs):
        try:
            x = np.atleast_1d(inputs.get(0, [0.0])).astype(float)
            gain = float(params.get('gain', 1.0))

            y = gain * x

            return {0: y, 'E': False}

        except Exception as e:
            logger.error(f"VectorGain error: {e}")
            return {0: np.array([0.0]), 'E': True, 'error': str(e)}
