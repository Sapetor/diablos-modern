import numpy as np
from blocks.base_block import BaseBlock
from blocks.param_templates import limit_params
from blocks.input_helpers import get_vector, clip_to_limits


class SaturationBlock(BaseBlock):
    """
    Saturates the input signal between lower and upper limits.
    """

    @property
    def block_name(self):
        return "Saturation"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def doc(self):
        return (
            "Limits the input signal to a specified range."
            "\n\nOutput:"
            "\n- Upper Limit if u > Upper Limit"
            "\n- Lower Limit if u < Lower Limit"
            "\n- u otherwise"
            "\n\nUsage:"
            "\nPrevents windup or limits actuator signals."
        )

    @property
    def params(self):
        return limit_params(
            min_doc="Lower saturation limit.",
            max_doc="Upper saturation limit."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw saturation/clipping icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Rails
        path.moveTo(0.1, 0.8)
        path.lineTo(0.9, 0.8)  # upper rail
        path.moveTo(0.1, 0.2)
        path.lineTo(0.9, 0.2)  # lower rail
        # Clipped signal
        path.moveTo(0.15, 0.5)
        path.quadTo(0.3, 0.2, 0.45, 0.2)
        path.lineTo(0.55, 0.2)
        path.quadTo(0.7, 0.8, 0.85, 0.8)
        return path

    def execute(self, time, inputs, params, **kwargs):
        u = get_vector(inputs, 0)
        u_sat = clip_to_limits(u, params)
        return {0: u_sat}

