
import numpy as np
from blocks.base_block import BaseBlock

class GainBlock(BaseBlock):
    """
    A block that scales an input by a factor.
    """

    @property
    def block_name(self):
        return "Gain"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "yellow"

    @property
    def doc(self):
        return "Scales an input by a factor."

    @property
    def params(self):
        return {
            "gain": {
                "type": "float",
                "default": 1.0,
                "doc": "The scaling factor."
            }
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        try:
            input_value = np.array(inputs[0], dtype=float)
            gain_value = np.array(params['gain'], dtype=float)
            return {0: np.dot(gain_value, input_value)}
        except (ValueError, TypeError):
            # In the future, this should raise a custom exception
            print(f"ERROR: Invalid input or gain type in gain block. Expected numeric.")
            return {'E': True}
