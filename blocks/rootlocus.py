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
            "\n\nAnalyzes the closed-loop poles of a system as a parameter varies (typically gain K)."
            "\n\nFeatures:"
            "\n- Connect to a Transfer Function or State Space block to define the system."
            "\n- Right-click the block and select 'Analysis > Root Locus' to generate the plot."
            "\n- Shows pole trajectories and stability boundaries."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        
        # Draw axes centered
        path.moveTo(0.5, 0.1)
        path.lineTo(0.5, 0.9)  # Imag axis
        path.moveTo(0.1, 0.5)
        path.lineTo(0.9, 0.5)  # Real axis
        
        # Draw some "branches"
        path.moveTo(0.3, 0.5) # Pole on left (stable)
        path.quadTo(0.3, 0.3, 0.5, 0.2) # Branch going to zero/asymptote
        
        path.moveTo(0.3, 0.5)
        path.quadTo(0.3, 0.7, 0.5, 0.8) # Mirror branch
        
        # Draw 'x' for a pole at 0.3, 0.5
        path.moveTo(0.28, 0.48); path.lineTo(0.32, 0.52)
        path.moveTo(0.32, 0.48); path.lineTo(0.28, 0.52)

        return path

    def execute(self, time, inputs, params):
        # RootLocus doesn't process data during simulation
        # It's used to generate static root locus plots via right-click menu
        # Return empty output dict to avoid breaking simulation
        return {}
