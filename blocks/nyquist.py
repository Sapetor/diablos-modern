from blocks.base_block import BaseBlock

class NyquistBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Nyquist"

    @property
    def category(self):
        return "Other"

    @property
    def color(self):
        return "purple"

    @property
    def params(self):
        return {
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Right-click to generate a Nyquist plot from a connected dynamic block."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def execute(self, time, inputs, params):
        # Nyquist doesn't process data during simulation
        return {}

    def draw_icon(self, block_rect):
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        
        # 1. Draw Axes (small cross)
        path.moveTo(0.2, 0.5); path.lineTo(0.8, 0.5)  # Real Axis
        path.moveTo(0.5, 0.2); path.lineTo(0.5, 0.8)  # Imag Axis
        
        # 2. Draw Spiral/Contour
        # Start near +infinity (right), spiral in
        path.moveTo(0.8, 0.4) 
        path.cubicTo(0.8, 0.9, 0.3, 0.9, 0.3, 0.5) # Bottom Loop
        path.cubicTo(0.3, 0.2, 0.6, 0.2, 0.6, 0.4) # Top Loop, spiraling in
        
        return path
