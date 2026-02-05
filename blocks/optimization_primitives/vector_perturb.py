"""
VectorPerturb Block

Perturbs x[index] by epsilon for finite difference gradient computation.
Used to build the finite difference structure for NumericalGradient.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class VectorPerturbBlock(BaseBlock):
    """
    Perturbs a single component of a vector by epsilon.

    Output: x_perturbed where x_perturbed[index] = x[index] + epsilon
    and all other components remain unchanged.

    Used for finite difference gradient computation.
    """

    @property
    def block_name(self):
        return "VectorPerturb"

    @property
    def category(self):
        return "Optimization Primitives"

    @property
    def color(self):
        return "cyan"

    @property
    def b_type(self):
        """Feedthrough block - direct input to output."""
        return 2

    @property
    def doc(self):
        return (
            "Perturbs x[index] by epsilon for finite difference gradient computation."
            "\n\nParameters:"
            "\n- index: Which component to perturb (0-indexed)"
            "\n- epsilon: Perturbation size (default: 1e-6)"
            "\n\nInput: Vector x"
            "\nOutput: Vector x with x[index] += epsilon"
        )

    @property
    def params(self):
        return {
            "index": {
                "type": "int",
                "default": 0,
                "doc": "Which component to perturb (0-indexed)"
            },
            "epsilon": {
                "type": "float",
                "default": 1e-6,
                "doc": "Perturbation size"
            },
        }

    @property
    def inputs(self):
        return [{"name": "x", "type": "vector"}]

    @property
    def outputs(self):
        return [{"name": "x_perturbed", "type": "vector"}]

    def execute(self, time, inputs, params, **kwargs):
        try:
            x = np.atleast_1d(inputs.get(0, [0.0])).copy().astype(float)
            index = int(params.get('index', 0))
            epsilon = float(params.get('epsilon', 1e-6))

            # Ensure index is within bounds
            if 0 <= index < len(x):
                x[index] += epsilon
            else:
                logger.warning(f"VectorPerturb: index {index} out of bounds for vector of length {len(x)}")

            return {0: x, 'E': False}

        except Exception as e:
            logger.error(f"VectorPerturb error: {e}")
            return {0: np.array([0.0]), 'E': True, 'error': str(e)}
