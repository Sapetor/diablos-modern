from blocks.base_block import BaseBlock
from lib import functions

class ExportBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Export"

    @property
    def category(self):
        return "Sinks"

    @property
    def color(self):
        return "red"

    @property
    def params(self):
        return {
            "str_name": {"default": "default", "type": "string"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Exports input signals to a file."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """Draw export/file icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Box
        path.moveTo(0.2, 0.2)
        path.lineTo(0.8, 0.2)
        path.lineTo(0.8, 0.8)
        path.lineTo(0.2, 0.8)
        path.lineTo(0.2, 0.2)
        # Arrow out
        path.moveTo(0.5, 0.5)
        path.lineTo(1.0, 0.5)
        path.moveTo(0.8, 0.3)
        path.lineTo(1.0, 0.5)
        path.lineTo(0.8, 0.7)
        return path

    def execute(self, time, inputs, params):
        return functions.export(time, inputs, params)

