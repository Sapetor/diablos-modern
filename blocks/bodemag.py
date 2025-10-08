from blocks.base_block import BaseBlock

class BodeMagBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "BodeMag"

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
        return "Right-click to generate a Bode magnitude plot from a connected Transfer Function block."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def execute(self, time, inputs, params):
        # BodeMag doesn't process data during simulation
        # It's used to generate static Bode plots via right-click menu
        return None
