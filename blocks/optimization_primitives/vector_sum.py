"""
VectorSum Block

Vector add/subtract: y = ±x1 ± x2 ...
Used for computing x_next = x_current + update in optimization.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class VectorSumBlock(BaseBlock):
    """
    Adds or subtracts multiple vector inputs.

    y = ±x1 ± x2 ± ... based on the signs parameter.

    Common uses:
    - x_next = x_current - alpha * gradient (signs = "+-")
    - x_next = x_current + velocity (signs = "++")
    """

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "VectorSum"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "green"

    @property
    def doc(self):
        return (
            "Adds or subtracts multiple vector inputs."
            "\n\nParameters:"
            "\n- signs: String of '+' and '-' for each input (e.g., '+-')"
            "\n\nInputs: x1, x2, ... (one per sign character)"
            "\nOutput: y = ±x1 ± x2 ± ..."
            "\n\nExample: signs='+-' computes y = x1 - x2"
        )

    @property
    def params(self):
        return {
            "signs": {
                "type": "string",
                "default": "++",
                "doc": "String of '+' and '-' for each input"
            },
        }

    @property
    def inputs(self):
        """Default inputs (used when block is first created)."""
        return [{"name": "x1", "type": "vector"}, {"name": "x2", "type": "vector"}]

    def get_inputs(self, params=None):
        """Get inputs dynamically based on signs parameter."""
        if params is None:
            params = self.params

        signs = params.get('signs', '++')
        if isinstance(signs, dict):
            signs = signs.get('default', '++')

        num_inputs = len(signs)
        return [{"name": f"x{i+1}", "type": "vector"} for i in range(num_inputs)]

    @property
    def outputs(self):
        return [{"name": "y", "type": "vector"}]

    def execute(self, time, inputs, params, **kwargs):
        try:
            signs = params.get('signs', '++')
            result = None

            for i in range(len(signs)):
                sign = signs[i] if i < len(signs) else '+'
                x = np.atleast_1d(inputs.get(i, [0.0])).astype(float)

                if result is None:
                    result = x.copy() if sign == '+' else -x.copy()
                else:
                    # Handle dimension mismatch by padding/truncating
                    if len(x) != len(result):
                        if len(x) < len(result):
                            x = np.pad(x, (0, len(result) - len(x)))
                        else:
                            x = x[:len(result)]

                    if sign == '+':
                        result = result + x
                    else:
                        result = result - x

            if result is None:
                result = np.array([0.0])

            return {0: result, 'E': False}

        except Exception as e:
            logger.error(f"VectorSum error: {e}")
            return {0: np.array([0.0]), 'E': True, 'error': str(e)}
