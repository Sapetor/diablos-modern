from blocks.base_block import BaseBlock
from lib import functions

class TransferFunctionBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "TranFn"

    @property
    def fn_name(self):
        return "transfer_function"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "numerator": {"default": [1.0], "type": "list"},
            "denominator": {"default": [1.0, 1.0], "type": "list"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Represents a linear time-invariant system as a transfer function."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """TranFn uses B(s)/A(s) text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        return functions.transfer_function(time, inputs, params, **kwargs)

