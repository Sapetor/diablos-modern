from blocks.base_block import BaseBlock

class RootLocusBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "RootLocus"

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
        return (
            "Root Locus Plotter."
            "\n\nAnalyzes the closed-loop poles of a system as a parameter varies."
            "\n\nUsage:"
            "\nConnect to a Transfer Function or State Space block."
            "\nRight-click to generate plot."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def execute(self, time, inputs, params):
        # RootLocus doesn't process data during simulation
        # It's used to generate static root locus plots via right-click menu
        # Return empty output dict to avoid breaking simulation
        return {}
