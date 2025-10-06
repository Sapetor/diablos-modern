from blocks.base_block import BaseBlock
import numpy as np

class ExponentialBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Exp"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "lime_green"

    @property
    def params(self):
        return {
            "a": {"default": 1.0, "type": "float"},
            "b": {"default": 1.0, "type": "float"},
        }

    @property
    def doc(self):
        return "Computes a * exp(b * x)"

    @property
    def inputs(self):
        return [{"name": "x", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "any"}]

    def execute(self, time, inputs, params):
        return {0: params['a'] * np.exp(params['b'] * inputs[0])}
