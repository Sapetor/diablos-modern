import numpy as np
from blocks.base_block import BaseBlock
from blocks.input_helpers import advance_sample_time, get_vector, safe_float, sample_due


class ZeroOrderHoldBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "ZeroOrderHold"

    @property
    def fn_name(self):
        return "zero_order_hold"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "sampling_time": {"default": 0.1, "type": "float"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return (
            "Zero-Order Hold (ZOH)."
            "\n\nSamples the input signal at a fixed rate and holds it constant between samples."
            "\n\nParameters:"
            "\n- Sampling Time: The period (in seconds) between samples."
            "\n\nUsage:"
            "\nConverts continuous signals to discrete (digital) steps."
            "\nModels triggers or ADCs."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw staircase/ZOH icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        path.moveTo(0.1, 0.8)
        path.lineTo(0.3, 0.8)
        path.lineTo(0.3, 0.5)
        path.lineTo(0.6, 0.5)
        path.lineTo(0.6, 0.2)
        path.lineTo(0.9, 0.2)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """
        Zero-Order Hold: Samples input at specified rate and holds value.
        """
        output_only = kwargs.get("output_only", False)

        if params.get("_init_start_", True):
            params["_init_start_"] = False
            params["_next_sample_time_"] = 0.0
            # Initialize held value with initial input if available, else 0
            params["_held_value_"] = get_vector(inputs, 0)

        # Get current held value
        held_val = np.atleast_1d(params.get("_held_value_", 0.0))

        # Check if it's time to sample
        sampling_time = safe_float(params.get("sampling_time", 0.1), 0.1)

        # A non-positive sample period has no valid sample grid; the schedule
        # loop below would spin forever trying to step past `time`.  Degrade to
        # a pass-through (sample on every call) instead of hanging the app.
        if not np.isfinite(sampling_time) or sampling_time <= 0.0:
            if output_only:
                return {0: held_val}
            params["_held_value_"] = get_vector(inputs, 0)
            params["_next_sample_time_"] = float(time)
            return {0: np.atleast_1d(params["_held_value_"])}

        if sample_due(time, params["_next_sample_time_"]):
            if not output_only:
                # Update held value
                params["_held_value_"] = get_vector(inputs, 0)

                # Schedule next sample
                advance_sample_time(params, "_next_sample_time_", time, sampling_time)

            return {0: np.atleast_1d(params["_held_value_"])}

        return {0: held_val}
