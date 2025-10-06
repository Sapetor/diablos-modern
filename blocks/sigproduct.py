
import numpy as np
from blocks.base_block import BaseBlock

class SigProductBlock(BaseBlock):
    """
    A block that calculates the element-wise product of its inputs.
    """

    @property
    def block_name(self):
        return "SgProd"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "lime_green"

    @property
    def doc(self):
        return "Calculates the element-wise product of its inputs."

    @property
    def params(self):
        return {}

    @property
    def inputs(self):
        return [{"name": "in1", "type": "any"}, {"name": "in2", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        try:
            mult = np.array(1.0, dtype=float)
            for input_value in inputs.values():
                mult *= np.array(input_value, dtype=float)
            return {0: mult}
        except (ValueError, TypeError):
            print(f"ERROR: Invalid input type in sigproduct block. Expected numeric.")
            return {'E': True}
