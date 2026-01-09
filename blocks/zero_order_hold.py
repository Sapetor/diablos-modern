from blocks.base_block import BaseBlock
from lib import functions

class ZeroOrderHoldBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "ZeroOrderHold"

    @property
    def fn_name(self):
        return "zero_order_hold"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "sampling_time": {"default": 0.1, "type": "float"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Samples the input at a specified rate and holds the value constant."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw staircase/ZOH icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(0.1, 0.8)
        path.lineTo(0.3, 0.8)
        path.lineTo(0.3, 0.5)
        path.lineTo(0.6, 0.5)
        path.lineTo(0.6, 0.2)
        path.lineTo(0.9, 0.2)
        return path

    def execute(self, time, inputs, params, **kwargs):
        return functions.zero_order_hold(time, inputs, params, **kwargs)

