from blocks.base_block import BaseBlock

class MuxBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Mux"

    @property
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

    @property
    def params(self):
        return {}

    @property
    def doc(self):
        return (
            "Multiplexer (Mux)."
            "\n\nCombines multiple scalar or vector signals into a single vector output."
            "\n\nParameters:"
            "\n- Inputs: Number of signals to combine."
            "\n\nUsage:"
            "\nUse to bundle signals for Scope plotting or bus routing."
        )

    @property
    def inputs(self):
        return [{"name": "in1", "type": "any"}, {"name": "in2", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw multiplexer icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Input lines
        path.moveTo(0.2, 0.3); path.lineTo(0.4, 0.3)
        path.moveTo(0.2, 0.7); path.lineTo(0.4, 0.7)
        # Main body (trapezoid)
        path.moveTo(0.4, 0.2)
        path.lineTo(0.8, 0.4)
        path.lineTo(0.8, 0.6)
        path.lineTo(0.4, 0.8)
        path.lineTo(0.4, 0.2)
        # Output
        path.moveTo(0.8, 0.5); path.lineTo(1.0, 0.5)
        return path

    def execute(self, time, inputs, params):
        return {0: list(inputs.values())}

