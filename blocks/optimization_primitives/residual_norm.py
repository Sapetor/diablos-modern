"""
ResidualNorm Block

Computes the norm of a vector for convergence checking.
‖F‖ is used to monitor convergence of optimization/root-finding algorithms.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class ResidualNormBlock(BaseBlock):
    """
    Computes the norm of a vector.

    Used for convergence monitoring in optimization:
    - ‖∇f‖ < tol indicates gradient descent convergence
    - ‖F(x)‖ < tol indicates root-finding convergence
    - ‖x_{k+1} - x_k‖ < tol indicates iterate convergence
    """

    @property
    def block_name(self):
        return "ResidualNorm"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "gray"

    @property
    def b_type(self):
        """Feedthrough block - direct input to output."""
        return 2

    @property
    def doc(self):
        return (
            "Computes the norm of a vector for convergence checking."
            "\n\nParameters:"
            "\n- norm_type: '2' (Euclidean), '1' (Manhattan), 'inf' (max abs)"
            "\n\nInput: Vector F (residual, gradient, or difference)"
            "\nOutput: Scalar ‖F‖"
            "\n\nUse with a Scope to monitor convergence."
        )

    @property
    def params(self):
        return {
            "norm_type": {
                "type": "choice",
                "default": "2",
                "options": ["1", "2", "inf"],
                "doc": "Norm type: 1 (Manhattan), 2 (Euclidean), inf (max)"
            },
        }

    @property
    def inputs(self):
        return [{"name": "F", "type": "vector"}]

    @property
    def outputs(self):
        return [{"name": "norm", "type": "float"}]

    def execute(self, time, inputs, params, **kwargs):
        try:
            F = np.atleast_1d(inputs.get(0, [0.0])).astype(float)
            norm_type = params.get('norm_type', '2')

            if norm_type == "1":
                norm = np.linalg.norm(F, ord=1)
            elif norm_type == "2":
                norm = np.linalg.norm(F, ord=2)
            elif norm_type == "inf":
                norm = np.linalg.norm(F, ord=np.inf)
            else:
                norm = np.linalg.norm(F, ord=2)

            return {0: float(norm), 'E': False}

        except Exception as e:
            logger.error(f"ResidualNorm error: {e}")
            return {0: 0.0, 'E': True, 'error': str(e)}
