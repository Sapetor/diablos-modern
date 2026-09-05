"""
VectorPerturb Block

Perturbs x[index] by epsilon for finite difference gradient computation.
Used to build the finite difference structure for NumericalGradient.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock
from blocks.input_helpers import get_vector

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
            "index": {"type": "int", "default": 0, "doc": "Which component to perturb (0-indexed)"},
            "epsilon": {"type": "float", "default": 1e-6, "doc": "Perturbation size"},
        }

    @property
    def inputs(self):
        return [{"name": "x", "type": "vector"}]

    @property
    def outputs(self):
        return [{"name": "x_perturbed", "type": "vector"}]

    def draw_icon(self, block_rect):
        """Draw a bracketed vector with one component nudged by epsilon."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        # Brackets
        path.moveTo(0.16, 0.10)
        path.lineTo(0.08, 0.10)
        path.lineTo(0.08, 0.90)
        path.lineTo(0.16, 0.90)
        path.moveTo(0.58, 0.10)
        path.lineTo(0.66, 0.10)
        path.lineTo(0.66, 0.90)
        path.lineTo(0.58, 0.90)
        # Components
        for row_y in (0.24, 0.50, 0.76):
            path.moveTo(0.20, row_y)
            path.lineTo(0.54, row_y)
        # Perturbation applied to one component
        path.moveTo(0.86, 0.72)
        path.lineTo(0.86, 0.26)
        path.moveTo(0.78, 0.36)
        path.lineTo(0.86, 0.26)
        path.lineTo(0.94, 0.36)
        return path

    def execute(self, time, inputs, params, **kwargs):
        try:
            x = get_vector(inputs, 0).astype(float)
            index = int(params.get("index", 0))
            epsilon = float(params.get("epsilon", 1e-6))

            # Ensure index is within bounds
            if 0 <= index < len(x):
                x[index] += epsilon
            else:
                logger.warning(
                    f"VectorPerturb: index {index} out of bounds for vector of length {len(x)}"
                )

            return {0: x, "E": False}

        except Exception as e:
            logger.error(f"VectorPerturb error: {e}")
            return {0: np.array([0.0]), "E": True, "error": str(e)}
