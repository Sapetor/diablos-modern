
import numpy as np
from blocks.base_block import BaseBlock

class RampBlock(BaseBlock):
    """
    A block that generates a ramp signal.
    """

    @property
    def block_name(self):
        return "Ramp"

    @property
    def category(self):
        return "Sources"

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return "Generates a ramp signal."

    @property
    def params(self):
        return {
            "slope": {"type": "float", "default": 1.0, "doc": "The slope of the ramp."},
            "delay": {"type": "float", "default": 0.0, "doc": "The delay of the ramp."}
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw ramp signal icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(0.1, 0.9)
        path.lineTo(0.9, 0.1)
        return path

    def execute(self, time, inputs, params):
        slope = float(params['slope'])
        delay = float(params['delay'])
        if slope == 0:
            return {0: np.array(0, dtype=float)}
        elif slope > 0:
            return {0: np.maximum(0, slope * (time - delay))}
        elif slope < 0:
            return {0: np.minimum(0, slope * (time - delay))}
