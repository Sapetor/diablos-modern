"""
ObjectiveFunction Block

Evaluates f(x) from a Python expression where x is a vector input.
Variables x1, x2, ... are mapped to components of the input vector.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class ObjectiveFunctionBlock(BaseBlock):
    """
    Evaluates an objective function f(x) from a Python expression.

    The input is a vector [x1, x2, ...] and the output is a scalar f(x).
    The expression can use variable names x1, x2, etc. to reference
    components of the input vector.
    """

    @property
    def block_name(self):
        return "ObjectiveFunction"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "purple"

    @property
    def doc(self):
        return (
            "Evaluates an objective function f(x) from a Python expression."
            "\n\nParameters:"
            "\n- expression: Python expression using x1, x2, ... (e.g., 'x1**2 + x2**2')"
            "\n- variables: Comma-separated variable names (e.g., 'x1,x2')"
            "\n\nInput: Vector x = [x1, x2, ...]"
            "\nOutput: Scalar f(x)"
            "\n\nAvailable functions: sin, cos, tan, exp, log, sqrt, abs, pi, e"
        )

    @property
    def params(self):
        return {
            "expression": {
                "type": "string",
                "default": "x1**2 + x2**2",
                "doc": "Python expression using x1, x2, ..."
            },
            "variables": {
                "type": "string",
                "default": "x1,x2",
                "doc": "Comma-separated variable names"
            },
        }

    @property
    def inputs(self):
        return [{"name": "x", "type": "vector"}]

    @property
    def outputs(self):
        return [{"name": "f", "type": "float"}]

    def execute(self, time, inputs, params, **kwargs):
        try:
            x = np.atleast_1d(inputs.get(0, [0.0]))
            variables = [v.strip() for v in params.get('variables', 'x1,x2').split(',')]
            expression = params.get('expression', 'x1**2 + x2**2')

            # Build context with x1, x2, ... mapped to vector components
            context = {
                "np": np,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
                "exp": np.exp, "log": np.log, "log10": np.log10,
                "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
                "pi": np.pi, "e": np.e,
                "t": time,
            }

            # Map variable names to vector components
            for i, var in enumerate(variables):
                context[var] = x[i] if i < len(x) else 0.0

            result = eval(expression, {"__builtins__": None}, context)
            return {0: float(result), 'E': False}

        except Exception as e:
            logger.error(f"ObjectiveFunction error: {e}")
            return {0: 0.0, 'E': True, 'error': str(e)}
