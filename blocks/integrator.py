from blocks.base_block import BaseBlock
import numpy as np
from lib import functions

class IntegratorBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Integrator"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "init_conds": {"default": 0.0, "type": "float"},
            "method": {"default": "SOLVE_IVP", "type": "string"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Integrates the input signal over time."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        return functions.integrator(time, inputs, params, **kwargs)
