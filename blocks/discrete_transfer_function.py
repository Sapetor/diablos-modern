from blocks.base_block import BaseBlock
from lib import functions

class DiscreteTransferFunctionBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "DiscreteTranFn"

    @property
    def fn_name(self):
        return "discrete_transfer_function"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "numerator": {"default": [1.0, 0.0], "type": "list"},
            "denominator": {"default": [1.0, -0.5], "type": "list"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Represents a discrete-time linear time-invariant system as a transfer function in z-domain."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    @property
    def b_type(self):
        """
        Determine block type based on properness.
        This is a default; it might be overridden during initialization based on actual params.
        Type 1: Strictly proper (memory)
        Type 2: Proper (direct feedthrough)
        """
        return 2 

    def draw_icon(self, block_rect):
        """DiscreteTranFn uses B(z)/A(z) text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        return functions.discrete_transfer_function(time, inputs, params, **kwargs)

