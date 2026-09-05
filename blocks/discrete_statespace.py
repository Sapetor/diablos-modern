from blocks.statespace_base import StateSpaceBaseBlock
import numpy as np


class DiscreteStateSpaceBlock(StateSpaceBaseBlock):
    """Discrete State-Space Model block with optional sampling time."""

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "DiscreteStateSpace"

    @property
    def category(self):
        return "Control"

    @property
    def fn_name(self):
        return "discrete_statespace"

    @property
    def params(self):
        return {
            "A": {"default": [[0.0]], "type": "list"},
            "B": {"default": [[1.0]], "type": "list"},
            "C": {"default": [[1.0]], "type": "list"},
            "D": {"default": [[0.0]], "type": "list"},
            "init_conds": {"default": [0.0], "type": "list"},
            "sampling_time": {
                "default": 0.0,
                "type": "float",
                "doc": (
                    "Sample period in seconds (0=inherit from the upstream rate, "
                    ">0=fixed rate). A discrete state recursion has no "
                    "continuous-time meaning, so -1 is not a useful setting here: "
                    "with no rate to inherit the block advances one sample per "
                    "solver step and its response then depends on the simulation "
                    "step size."
                ),
            },
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def requires_sample_time(self):
        """Pure discrete state recursion — undefined without a resolved rate."""
        return True

    @property
    def doc(self):
        return (
            "Discrete State-Space Model."
            "\n\nx[k+1] = Ax[k] + Bu[k]"
            "\ny[k] = Cx[k] + Du[k]"
            "\n\nParameters:"
            "\n- A, B, C, D: Discrete system matrices."
            "\n- Sampling Time: Execution rate."
            "\n\nUsage:"
            "\nDigital Modern Control (MIMO)."
        )

    @property
    def b_type(self):
        """Block type: 1=strictly proper (memory), 2=proper (direct feedthrough)."""
        return 2

    def draw_icon(self, block_rect):
        """Draw a bracketed coefficient matrix beside a discrete "z"."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        # Left bracket
        path.moveTo(0.26, 0.12)
        path.lineTo(0.16, 0.12)
        path.lineTo(0.16, 0.88)
        path.lineTo(0.26, 0.88)
        # Right bracket
        path.moveTo(0.60, 0.12)
        path.lineTo(0.70, 0.12)
        path.lineTo(0.70, 0.88)
        path.lineTo(0.60, 0.88)
        # Matrix entries
        for row_y in (0.34, 0.66):
            path.moveTo(0.26, row_y)
            path.lineTo(0.38, row_y)
            path.moveTo(0.48, row_y)
            path.lineTo(0.60, row_y)
        # "z" marking the discrete-time recursion
        path.moveTo(0.80, 0.30)
        path.lineTo(0.98, 0.30)
        path.lineTo(0.80, 0.62)
        path.lineTo(0.98, 0.62)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Execute discrete state-space block with optional sampling time."""
        output_only = kwargs.get("output_only", False)

        if params.get("_init_start_", True):
            params["_init_start_"] = False

            # Validate matrices (already discrete, no conversion needed)
            result = self._validate_state_space_matrices(
                params["A"], params["B"], params["C"], params["D"]
            )
            if isinstance(result, dict):
                return result
            A, B, C, D, n, m, p = result

            params["_Ad_"] = A
            params["_Bd_"] = B
            params["_Cd_"] = C
            params["_Dd_"] = D
            params["_x_"] = self._initialize_state_vector(n, params.get("init_conds", [0.0]))
            params["_n_states_"] = n
            params["_n_inputs_"] = m
            params["_n_outputs_"] = p
            params["_next_sample_time_"] = 0.0
            params["_held_output_"] = 0.0 if p == 1 else np.zeros(p)

        # Between sample instants the block holds its last output.
        if not self._sample_due(time, params):
            return {0: params.get("_held_output_", 0.0), "E": False}

        # Read the input, form y = Cx + Du, then advance the state.
        y_val, err = self._step(inputs, params, output_only)
        if err is not None:
            return err

        params["_held_output_"] = y_val

        if not output_only:
            self._schedule_next_sample(time, params)

        return {0: y_val, "E": False}
