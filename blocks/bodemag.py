from blocks.base_block import BaseBlock


class BodeMagBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "BodeMag"

    @property
    def category(self):
        return "Analysis"

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
        return "Right-click to generate a Bode magnitude plot from a connected Transfer Function block."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """Draw a magnitude asymptote rolling off over log axes."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        # Axes
        path.moveTo(0.1, 0.9)
        path.lineTo(0.9, 0.9)
        path.moveTo(0.1, 0.9)
        path.lineTo(0.1, 0.1)
        # Magnitude asymptote with a single break
        path.moveTo(0.1, 0.4)
        path.lineTo(0.4, 0.4)
        path.lineTo(0.6, 0.7)
        path.lineTo(0.9, 0.7)
        return path

    def execute(self, time, inputs, params, **kwargs):
        # BodeMag doesn't process data during simulation
        # It's used to generate static Bode plots via right-click menu
        # Return empty output dict to avoid breaking simulation
        return {}
