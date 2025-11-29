from blocks.base_block import BaseBlock
from lib import functions

class DiscreteStateSpaceBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "DiscreteStateSpace"

    @property
    def fn_name(self):
        return "discrete_statespace"

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
        return """Discrete State-Space representation: x[k+1] = Ax[k] + Bu[k], y[k] = Cx[k] + Du[k]

        A: State matrix (n×n)
        B: Input matrix (n×m)
        C: Output matrix (p×n)
        D: Feedthrough matrix (p×m)

        where n = number of states, m = number of inputs, p = number of outputs
        """

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    @property
    def b_type(self):
        """
        Determine block type based on properness.
        This is a default; it might be overridden during initialization based on actual params.
        Type 1: Strictly proper (memory)
        Type 2: Proper (direct feedthrough)
        """
        return 2

    def execute(self, time, inputs, params, **kwargs):
        return functions.discrete_statespace(time, inputs, params, **kwargs)
