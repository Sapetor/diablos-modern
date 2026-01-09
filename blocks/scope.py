from blocks.base_block import BaseBlock
from lib import functions

class ScopeBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Scope"

    @property
    def category(self):
        return "Sinks"

    @property
    def color(self):
        return "red"

    @property
    def params(self):
        return {
            "labels": {"default": "default", "type": "string"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Displays input signals on a plot."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """Draw oscilloscope icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(0.1, 0.9)
        path.lineTo(0.9, 0.9)  # x-axis
        path.moveTo(0.1, 0.9)
        path.lineTo(0.1, 0.1)  # y-axis
        path.moveTo(0.1, 0.6)
        path.quadTo(0.3, 0.2, 0.5, 0.6)
        path.quadTo(0.7, 1.0, 0.9, 0.6)
        return path

    def execute(self, time, inputs, params):
        return functions.scope(time, inputs, params)

