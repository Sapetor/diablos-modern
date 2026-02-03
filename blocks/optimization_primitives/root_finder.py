"""
RootFinder Block

Computes one Newton step for solving F(x) = 0.
x_next = x - J^{-1} F(x)

Can also be used with expressions for automatic Jacobian approximation.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class RootFinderBlock(BaseBlock):
    """
    Computes one Newton step for solving F(x) = 0.

    Newton's method: x_{k+1} = x_k - J(x_k)^{-1} F(x_k)

    This block can work in two modes:
    1. Expression mode: Provide F as expressions, Jacobian computed numerically
    2. Input mode: Receive F(x) and J(x) as inputs from other blocks
    """

    @property
    def block_name(self):
        return "RootFinder"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "red"

    @property
    def doc(self):
        return (
            "Computes one Newton step for solving F(x) = 0."
            "\n\nParameters:"
            "\n- expressions: Comma-separated F expressions (e.g., 'x1**2+x2-1, x1+x2**2-1')"
            "\n- variables: Comma-separated variable names (e.g., 'x1,x2')"
            "\n- epsilon: Perturbation for numerical Jacobian (default: 1e-6)"
            "\n- damping: Damping factor for step (default: 1.0)"
            "\n\nInput: Current x vector"
            "\nOutput: Next x vector (one Newton step)"
        )

    @property
    def params(self):
        return {
            "expressions": {
                "type": "string",
                "default": "x1**2 + x2 - 1, x1 + x2**2 - 1",
                "doc": "Comma-separated F expressions"
            },
            "variables": {
                "type": "string",
                "default": "x1,x2",
                "doc": "Comma-separated variable names"
            },
            "epsilon": {
                "type": "float",
                "default": 1e-6,
                "doc": "Perturbation for numerical Jacobian"
            },
            "damping": {
                "type": "float",
                "default": 1.0,
                "doc": "Damping factor (0 < damping <= 1)"
            },
        }

    @property
    def inputs(self):
        return [{"name": "x", "type": "vector"}]

    @property
    def outputs(self):
        return [{"name": "x_next", "type": "vector"}]

    def _eval_F(self, x, expressions, variables):
        """Evaluate F(x) from expressions."""
        context = {
            "np": np,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
            "abs": np.abs, "pi": np.pi, "e": np.e,
        }

        for i, var in enumerate(variables):
            context[var] = x[i] if i < len(x) else 0.0

        F = np.zeros(len(expressions))
        for i, expr in enumerate(expressions):
            F[i] = eval(expr.strip(), {"__builtins__": None}, context)

        return F

    def _compute_jacobian(self, x, expressions, variables, epsilon):
        """Compute Jacobian numerically using finite differences."""
        n = len(variables)
        m = len(expressions)
        J = np.zeros((m, n))

        F_center = self._eval_F(x, expressions, variables)

        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += epsilon
            F_plus = self._eval_F(x_plus, expressions, variables)
            J[:, j] = (F_plus - F_center) / epsilon

        return J

    def execute(self, time, inputs, params, **kwargs):
        try:
            x = np.atleast_1d(inputs.get(0, [0.0, 0.0])).astype(float)

            expressions = [e.strip() for e in params.get('expressions', 'x1**2+x2-1,x1+x2**2-1').split(',')]
            variables = [v.strip() for v in params.get('variables', 'x1,x2').split(',')]
            epsilon = float(params.get('epsilon', 1e-6))
            damping = float(params.get('damping', 1.0))

            # Ensure x has correct dimension
            if len(x) < len(variables):
                x = np.pad(x, (0, len(variables) - len(x)))

            # Evaluate F(x)
            F = self._eval_F(x, expressions, variables)

            # Compute Jacobian
            J = self._compute_jacobian(x, expressions, variables, epsilon)

            # Solve J * delta = -F for delta
            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                delta = -np.linalg.pinv(J) @ F

            # Apply damping
            x_next = x + damping * delta

            return {0: x_next, 'E': False}

        except Exception as e:
            logger.error(f"RootFinder error: {e}")
            return {0: np.atleast_1d(inputs.get(0, [0.0])), 'E': True, 'error': str(e)}
