import numpy as np
from blocks.base_block import BaseBlock


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
        return "Clips the input signal to specified min/max limits."

    @property
    def params(self):
        return {
            "min": {"type": "float", "default": -np.inf, "doc": "Lower saturation limit."},
            "max": {"type": "float", "default": np.inf, "doc": "Upper saturation limit."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        u = np.array(inputs[0], dtype=float)
        u_sat = np.clip(u, params["min"], params["max"])
        return {0: u_sat}
