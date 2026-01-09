from blocks.base_block import BaseBlock
from lib import functions

class StateSpaceBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "StateSpace"

    @property
    def fn_name(self):
        return "statespace"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "A": {"default": [[0.0]], "type": "list"},  # State matrix (n×n)
            "B": {"default": [[1.0]], "type": "list"},  # Input matrix (n×m)
            "C": {"default": [[1.0]], "type": "list"},  # Output matrix (p×n)
            "D": {"default": [[0.0]], "type": "list"},  # Feedthrough matrix (p×m)
            "init_conds": {"default": [0.0], "type": "list"},  # Initial state vector (n×1)
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return """State-Space representation: dx/dt = Ax + Bu, y = Cx + Du

        A: State matrix (n×n)
        B: Input matrix (n×m)
        C: Output matrix (p×n)
        D: Feedthrough matrix (p×m)

        where n = number of states, m = number of inputs, p = number of outputs

        Example single-input single-output:
        A = [[0, 1], [-2, -3]]
        B = [[0], [1]]
        C = [[1, 0]]
        D = [[0]]
        """

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """StateSpace uses complex rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        return functions.statespace(time, inputs, params, **kwargs)

