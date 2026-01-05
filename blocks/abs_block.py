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
        return "Outputs the absolute value of the input signal."

    @property
    def params(self):
        return {}

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        input_value = np.atleast_1d(inputs.get(0, 0))
        return {0: np.abs(input_value)}
