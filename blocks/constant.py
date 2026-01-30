import numpy as np
from blocks.base_block import BaseBlock


class ConstantBlock(BaseBlock):
    """
    Outputs a constant value at every time step.
    """

    @property
    def block_name(self):
        return "Constant"

    @property
    def category(self):
        return "Sources"

    @property
    def color(self):
        return "green"

    @property
    def doc(self):
        return (
            "Outputs a constant value."
            "\n\nParameters:"
            "\n- Value: The constant output value (scalar or vector)."
            "\n\nUsage:"
            "\nUseful for setpoints, constant parameters, or enabling blocks."
            "\nTo create a vector, use [v1, v2, ...]."
        )

    @property
    def params(self):
        return {
            "value": {"type": "float", "default": 1.0, "doc": "Constant output value."},
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw constant value icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Horizontal line representing constant
        path.moveTo(0.1, 0.5)
        path.lineTo(0.9, 0.5)
        return path

    def execute(self, time, inputs, params, **kwargs):
        value = params.get("value", 1.0)
        return {0: np.atleast_1d(value), 'E': False}

