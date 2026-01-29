
import numpy as np
from blocks.base_block import BaseBlock

class SigProductBlock(BaseBlock):
    """
    A block that calculates the element-wise product of its inputs.
    """

    @property
    def block_name(self):
        return "SgProd"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "lime_green"

    @property
    def doc(self):
        return (
            "Computes the product of input signals."
            "\n\nOperation:"
            "\ny = u1 * u2 * ... * un (Element-wise multiplication)."
            "\n\nParameters:"
            "\n- Inputs: Number of input ports to multiply."
            "\n\nUsage:"
            "\nUsed for modulation, mixing, or non-linear scaling."
        )

    @property
    def params(self):
        return {}

    @property
    def inputs(self):
        return [{"name": "in1", "type": "any"}, {"name": "in2", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw multiplication X symbol in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(0.2, 0.2)
        path.lineTo(0.8, 0.8)
        path.moveTo(0.2, 0.8)
        path.lineTo(0.8, 0.2)
        return path

    def execute(self, time, inputs, params):
        try:
            mult = np.array(1.0, dtype=float)
            for input_value in inputs.values():
                mult *= np.array(input_value, dtype=float)
            return {0: mult}
        except (ValueError, TypeError):
            print(f"ERROR: Invalid input type in sigproduct block. Expected numeric.")
            return {'E': True}

