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
        return "Outputs a constant value. Useful for setpoints, constants, and fixed parameters."

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

    def execute(self, time, inputs, params):
        value = params.get("value", 1.0)
        return {0: np.atleast_1d(value)}
