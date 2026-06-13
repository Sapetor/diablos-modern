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
        return (
            "Exponential Signal."
            "\n\ny(t) = Amplitude * exp(Rate * t)"
            "\n\nParameters:"
            "\n- Amplitude: Initial value."
            "\n- Rate: Growth (+) or Decay (-) constant."
            "\n\nUsage:"
            "\nTransient analysis or unstable system simulation."
        )

    @property
    def inputs(self):
        return [{"name": "x", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        try:
            x = inputs.get(0, 0.0)
            a = params.get('a', 1.0)
            b = params.get('b', 1.0)
            return {0: a * np.exp(np.clip(b * x, -700, 700))}
        except Exception as e:
            return {'E': True, 'error': str(e)}
