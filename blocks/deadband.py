import numpy as np
from blocks.base_block import BaseBlock


class DeadbandBlock(BaseBlock):
    """
    Zeroes the signal inside a symmetric deadband around center; passes through otherwise.
    """

    @property
    def block_name(self):
        return "Deadband"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def doc(self):
        return "Suppress small signals within +/- deadband around center."

    @property
    def params(self):
        return {
            "deadband": {"type": "float", "default": 0.1, "doc": "Half-width of deadband."},
            "center": {"type": "float", "default": 0.0, "doc": "Center value of deadband."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        u = np.array(inputs[0], dtype=float)
        center = float(params.get("center", 0.0))
        band = float(params.get("deadband", 0.0))
        out = u.copy()
        mask = np.abs(u - center) <= band
        out[mask] = center
        return {0: out}
