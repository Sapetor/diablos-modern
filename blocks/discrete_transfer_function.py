from blocks.statespace_base import StateSpaceBaseBlock
import numpy as np
from scipy import signal


class DiscreteTransferFunctionBlock(StateSpaceBaseBlock):
    """Discrete Transfer Function block in z-domain."""

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "DiscreteTranFn"

    @property
    def category(self):
        return "Control"

    @property
    def fn_name(self):
        return "discrete_transfer_function"

    @property
    def params(self):
        return {
            "numerator": {"default": [1.0, 0.0], "type": "list"},
            "denominator": {"default": [1.0, -0.5], "type": "list"},
            "sampling_time": {
                "default": 0.0,
                "type": "float",
                "doc": (
                    "Sample period in seconds (0=inherit from the upstream rate, "
                    ">0=fixed rate). A z-domain block has no continuous-time "
                    "meaning, so -1 is not a useful setting here: with no rate to "
                    "inherit the block advances one sample per solver step and its "
                    "response then depends on the simulation step size."
                ),
            },
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def requires_sample_time(self):
        """Pure z-domain recursion — undefined without a resolved rate."""
        return True

    @property
    def doc(self):
        return "Represents a discrete-time linear time-invariant system as a transfer function in z-domain."

    @property
    def b_type(self):
        """Block type: 1=strictly proper (memory), 2=proper (direct feedthrough)."""
        return 2

    def draw_icon(self, block_rect):
        """DiscreteTranFn uses B(z)/A(z) text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        """Execute discrete transfer function with optional sampling time."""
        output_only = kwargs.get("output_only", False)

        if params.get("_init_start_", True):
            params["_init_start_"] = False
            num = np.array(params["numerator"], dtype=float)
            den = np.array(params["denominator"], dtype=float)

            # Convert to state-space (already discrete)
            try:
                A, B, C, D = signal.tf2ss(num, den)
            except Exception as e:
                return {"E": True, "error": f"Error in tf2ss conversion: {e}"}

            params["_Ad_"] = A
            params["_Bd_"] = B
            params["_Cd_"] = C
            params["_Dd_"] = D

            # Initialize state vector
            n = A.shape[0]
            params["_x_"] = self._initialize_state_vector(n, params.get("init_conds", [0.0]))
            params["_n_states_"] = n
            params["_n_inputs_"] = 1
            params["_n_outputs_"] = 1
            params["_next_sample_time_"] = 0.0
            params["_held_output_"] = 0.0

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
