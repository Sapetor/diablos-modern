"""
NumericalGradient Block

Computes gradient from externally-provided f(x), f(x+εe_i) values.
User builds the finite difference structure with VectorPerturb + ObjectiveFunction blocks,
then feeds f_center and f_plus_i values into this block.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class NumericalGradientBlock(BaseBlock):
    """
    Computes the gradient of f using finite differences.

    Receives f(x) at center point and f(x + ε*e_i) for each dimension.
    Outputs the gradient vector ∇f.

    Forward difference: ∇f[i] = (f(x + ε*e_i) - f(x)) / ε
    Central difference: ∇f[i] = (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)
    """

    def __init__(self):
        super().__init__()
        self._cached_dimension = None

    @property
    def block_name(self):
        return "NumericalGradient"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "orange"

    @property
    def b_type(self):
        """Feedthrough block - direct input to output."""
        return 2

    @property
    def doc(self):
        return (
            "Computes gradient from finite difference inputs."
            "\n\nParameters:"
            "\n- dimension: Number of variables (creates that many f_plus inputs)"
            "\n- epsilon: Perturbation size used (must match VectorPerturb blocks)"
            "\n- method: 'forward' or 'central' difference"
            "\n\nInputs:"
            "\n- f_center: f(x) at the center point"
            "\n- f_plus_0, f_plus_1, ...: f(x + ε*e_i) for each dimension"
            "\n- (For central: also f_minus_0, f_minus_1, ...)"
            "\n\nOutput: Gradient vector ∇f"
        )

    @property
    def params(self):
        return {
            "dimension": {
                "type": "int",
                "default": 2,
                "doc": "Number of variables"
            },
            "epsilon": {
                "type": "float",
                "default": 1e-6,
                "doc": "Perturbation size"
            },
            "method": {
                "type": "choice",
                "default": "forward",
                "options": ["forward", "central"],
                "doc": "Finite difference method"
            },
        }

    @property
    def inputs(self):
        """Default inputs - actual inputs depend on dimension parameter."""
        return [
            {"name": "f_center", "type": "float"},
            {"name": "f_plus_0", "type": "float"},
            {"name": "f_plus_1", "type": "float"},
        ]

    def get_inputs(self, params=None):
        """Get inputs dynamically based on dimension and method."""
        if params is None:
            params = self.params

        dimension = params.get('dimension', 2)
        if isinstance(dimension, dict):
            dimension = dimension.get('default', 2)
        dimension = int(dimension)

        method = params.get('method', 'forward')
        if isinstance(method, dict):
            method = method.get('default', 'forward')

        inputs = [{"name": "f_center", "type": "float"}]

        for i in range(dimension):
            inputs.append({"name": f"f_plus_{i}", "type": "float"})

        if method == "central":
            for i in range(dimension):
                inputs.append({"name": f"f_minus_{i}", "type": "float"})

        return inputs

    @property
    def outputs(self):
        return [{"name": "grad", "type": "vector"}]

    def _to_scalar(self, val, default=0.0):
        """Convert input value to scalar, handling arrays and numpy types."""
        if val is None:
            return default
        # Handle numpy arrays (including 0-d arrays)
        arr = np.atleast_1d(val).ravel()
        return float(arr[0]) if len(arr) > 0 else default

    def execute(self, time, inputs, params, **kwargs):
        try:
            dimension = int(params.get('dimension', 2))
            epsilon = float(params.get('epsilon', 1e-6))
            method = params.get('method', 'forward')

            f_center = self._to_scalar(inputs.get(0), 0.0)
            grad = np.zeros(dimension)

            if method == "forward":
                for i in range(dimension):
                    f_plus = self._to_scalar(inputs.get(i + 1), f_center)
                    grad[i] = (f_plus - f_center) / epsilon
            else:  # central
                for i in range(dimension):
                    f_plus = self._to_scalar(inputs.get(i + 1), f_center)
                    f_minus = self._to_scalar(inputs.get(dimension + 1 + i), f_center)
                    grad[i] = (f_plus - f_minus) / (2 * epsilon)

            return {0: grad, 'E': False}

        except Exception as e:
            logger.error(f"NumericalGradient error: {e}")
            dimension = int(params.get('dimension', 2))
            return {0: np.zeros(dimension), 'E': True, 'error': str(e)}
