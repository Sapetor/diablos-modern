
import logging
import ast
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class MatrixGainBlock(BaseBlock):
    """
    A gain block that accepts scalar, vector, matrix values or workspace variables.

    The gain parameter is a text field, so you can enter:
    - A scalar: 2.5
    - A vector: [1, 2, 3]
    - A matrix: [[1, 0], [0, 2]]
    - A workspace variable name: K

    Operations:
    - Scalar: y = K * u (element-wise)
    - Vector: y = K * u (element-wise, same length)
    - Matrix: y = K @ u (matrix-vector multiplication)
    """

    @property
    def block_name(self):
        return "MatrixGain"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "yellow"

    @property
    def doc(self):
        return (
            "Scales the input by a gain that can be a scalar, vector, or matrix."
            "\n\nAccepts workspace variable names (e.g. K) or direct values."
            "\n\nOperations:"
            "\n- Scalar: y = K * u"
            "\n- Vector: y = K .* u (element-wise)"
            "\n- Matrix: y = K @ u (matrix-vector product)"
            "\n\nExamples:"
            "\n  2.5              (scalar)"
            "\n  [1, 2, 3]       (vector)"
            "\n  [[1, 0], [0, 2]] (matrix)"
            "\n  K                (workspace variable)"
        )

    @property
    def params(self):
        return {
            "gain": {
                "type": "string",
                "default": "1.0",
                "doc": "Gain: scalar, vector, matrix, or workspace variable name."
            }
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    @property
    def use_port_grid_snap(self):
        return False

    def draw_icon(self, block_rect):
        """Triangle with 'M' subscript — handled by block renderer for triangle shape."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        try:
            u = np.atleast_1d(inputs.get(0, 0)).astype(float)

            K_raw = params.get('gain', '1.0')

            if isinstance(K_raw, str):
                try:
                    K = np.array(ast.literal_eval(K_raw), dtype=float)
                except (ValueError, SyntaxError):
                    K = np.array([float(K_raw)], dtype=float)
            else:
                K = np.atleast_1d(np.array(K_raw, dtype=float))

            if K.ndim == 2:
                u = u.flatten()
                if K.shape[1] != len(u):
                    if len(u) < K.shape[1]:
                        u = np.pad(u, (0, K.shape[1] - len(u)))
                    else:
                        u = u[:K.shape[1]]
                y = K @ u
            elif K.ndim == 1 and len(K) > 1:
                if len(K) == len(u):
                    y = K * u
                elif len(K) == 1:
                    y = K[0] * u
                else:
                    min_len = min(len(K), len(u))
                    y = K[:min_len] * u[:min_len]
            else:
                y = K.flatten()[0] * u

            return {0: y}
        except (ValueError, TypeError) as e:
            logger.error(f"MatrixGain block error: {str(e)}")
            return {'E': True, 'error': str(e)}
