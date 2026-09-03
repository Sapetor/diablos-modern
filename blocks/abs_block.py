import numpy as np
from blocks.base_block import BaseBlock


class AbsBlock(BaseBlock):
    """
    Computes the absolute value of the input signal.
    """

    @property
    def block_name(self):
        return "Abs"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return (
            "Absolute Value."
            "\n\nComputes the absolute value of the input signal."
            "\ny = |u|"
            "\n\nUsage:"
            "\nUsed in magnitude calculations, rectifiers, or error metrics."
        )

    @property
    def params(self):
        return {}

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw |u|: the V-shaped rectifier characteristic over a baseline."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        # Baseline (input axis)
        path.moveTo(0.05, 0.82)
        path.lineTo(0.95, 0.82)
        # y = |u|
        path.moveTo(0.16, 0.12)
        path.lineTo(0.50, 0.82)
        path.lineTo(0.84, 0.12)
        return path

    def execute(self, time, inputs, params, **kwargs):
        input_value = np.atleast_1d(inputs.get(0, 0))
        return {0: np.abs(input_value)}
