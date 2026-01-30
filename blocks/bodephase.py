from blocks.base_block import BaseBlock

class BodePhaseBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "BodePhase"

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
        return "Right-click to generate a Bode Phase plot from a connected dynamic block."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def execute(self, time, inputs, params, **kwargs):
        # BodePhase doesn't process data during simulation
        return {}

    def draw_icon(self, block_rect):
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        
        # 1. Draw Axes (L-shape for plot)
        path.moveTo(0.15, 0.15); path.lineTo(0.15, 0.85); path.lineTo(0.85, 0.85)
        
        # 2. Draw Phase Curve (High to Low transition)
        path.moveTo(0.25, 0.25) # Start high freq/phase
        path.cubicTo(0.45, 0.25, 0.55, 0.75, 0.75, 0.75) # S-curve transition
        
        return path
