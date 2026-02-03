"""
LinearSystemSolver Block

Solves the linear system Ax = b for x.
Used in Newton's method and other optimization algorithms.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class LinearSystemSolverBlock(BaseBlock):
    """
    Solves the linear system Ax = b.

    Methods:
    - 'direct': Uses numpy.linalg.solve (LU decomposition)
    - 'lstsq': Uses numpy.linalg.lstsq (least squares, works for non-square)
    - 'pinv': Uses pseudo-inverse A⁺b
    """

    @property
    def block_name(self):
        return "LinearSystemSolver"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "magenta"

    @property
    def doc(self):
        return (
            "Solves the linear system Ax = b for x."
            "\n\nParameters:"
            "\n- method: 'direct' (LU), 'lstsq' (least squares), or 'pinv' (pseudo-inverse)"
            "\n- regularization: Small value added to diagonal for stability (default: 0)"
            "\n\nInputs:"
            "\n- A: Matrix (flattened, with dimension parameter)"
            "\n- b: Vector"
            "\n\nOutput: Solution vector x"
        )

    @property
    def params(self):
        return {
            "method": {
                "type": "choice",
                "default": "direct",
                "options": ["direct", "lstsq", "pinv"],
                "doc": "Solution method"
            },
            "dimension": {
                "type": "int",
                "default": 2,
                "doc": "System dimension (for reshaping A)"
            },
            "regularization": {
                "type": "float",
                "default": 0.0,
                "doc": "Tikhonov regularization (added to diagonal)"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "A", "type": "matrix"},
            {"name": "b", "type": "vector"},
        ]

    @property
    def outputs(self):
        return [{"name": "x", "type": "vector"}]

    def execute(self, time, inputs, params, **kwargs):
        try:
            method = params.get('method', 'direct')
            dimension = int(params.get('dimension', 2))
            regularization = float(params.get('regularization', 0.0))

            # Get A matrix - could be flattened or 2D
            A_input = inputs.get(0, np.eye(dimension))
            A = np.atleast_1d(A_input).astype(float)

            # Reshape if flattened
            if A.ndim == 1:
                n = int(np.sqrt(len(A)))
                if n * n == len(A):
                    A = A.reshape(n, n)
                else:
                    A = A.reshape(dimension, -1)

            # Get b vector
            b = np.atleast_1d(inputs.get(1, np.zeros(dimension))).astype(float)

            # Apply regularization if needed
            if regularization > 0:
                A = A + regularization * np.eye(A.shape[0])

            # Solve based on method
            if method == "direct":
                try:
                    x = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    # Fall back to lstsq if singular
                    x, *_ = np.linalg.lstsq(A, b, rcond=None)
            elif method == "lstsq":
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
            else:  # pinv
                x = np.linalg.pinv(A) @ b

            return {0: x, 'E': False}

        except Exception as e:
            logger.error(f"LinearSystemSolver error: {e}")
            dimension = int(params.get('dimension', 2))
            return {0: np.zeros(dimension), 'E': True, 'error': str(e)}
