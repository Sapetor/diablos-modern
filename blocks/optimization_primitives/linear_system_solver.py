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
    def b_type(self):
        """Feedthrough block - direct input to output."""
        return 2

    @property
    def doc(self):
        return (
            "Solves the linear system Ax = b for x."
            "\n\nParameters:"
            "\n- method: 'direct' (LU), 'lstsq' (least squares), or 'pinv' (pseudo-inverse)"
            "\n- dimension: System size n (for n×n matrix A)"
            "\n- regularization: Small value added to diagonal for stability (default: 0)"
            "\n\nInputs:"
            "\n- A: Matrix - accepts nested [[1,2],[3,4]] or flattened [1,2,3,4]"
            "\n- b: Vector - accepts nested [[1],[2]] or flat [1,2]"
            "\n\nOutput: Solution vector x"
            "\n\nExample (2×2 system):"
            "\n  A = [[2,1],[1,3]] or [2,1,1,3]"
            "\n  b = [5,7] → x ≈ [1.6, 1.8]"
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

            # Get A matrix - handles nested [[1,2],[3,4]] or flattened [1,2,3,4]
            A_input = inputs.get(0, np.eye(dimension))
            A = np.array(A_input, dtype=float)

            # Handle nested list input like [[1,2],[3,4]]
            if A.ndim == 2:
                pass  # Already 2D, use as-is
            elif A.ndim == 1:
                # Flattened - reshape to square matrix
                n = int(np.sqrt(len(A)))
                if n * n == len(A):
                    A = A.reshape(n, n)
                else:
                    A = A.reshape(dimension, -1)
            else:
                # Unexpected shape, flatten and reshape
                A = A.flatten().reshape(dimension, dimension)

            # Get b vector - handles nested [[1],[2]] or flat [1,2]
            b_input = inputs.get(1, np.zeros(dimension))
            b = np.array(b_input, dtype=float).flatten()  # Flatten handles both cases

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
