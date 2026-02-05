import numpy as np
from blocks.base_block import BaseBlock
from blocks.param_templates import slew_rate_params, init_flag_param
from blocks.input_helpers import get_vector, InitStateManager


class RateLimiterBlock(BaseBlock):
    """
    Limits the rate of change (slew rate) of the input signal.
    """

    @property
    def block_name(self):
        return "RateLimiter"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def doc(self):
        return (
            "Rate Limiter."
            "\n\nLimits the rate of change (slope) of the input signal."
            "\n\nParameters:"
            "\n- Rising Slew Rate: Max positive slope (dy/dt)."
            "\n- Falling Slew Rate: Max negative slope (dy/dt) (usually negative)."
            "\n\nUsage:"
            "\nPrevents abrupt changes in control signals or models actuator speed limits."
            "\nUseful for smoothing setpoints."
        )

    @property
    def params(self):
        return {
            **slew_rate_params(),
            **init_flag_param(),
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw rate limiter icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Step response with slew limit
        path.moveTo(0.15, 0.75)
        path.lineTo(0.35, 0.75)
        path.lineTo(0.65, 0.25)  # Slew-limited ramp
        path.lineTo(0.85, 0.25)
        return path

    def execute(self, time, inputs, params, **kwargs):
        dt = float(params.get("dtime", 0.01))
        u = get_vector(inputs, 0)

        init_mgr = InitStateManager(params)
        if init_mgr.needs_init():
            params["_prev"] = u
            init_mgr.mark_initialized()
            return {0: u}

        prev = np.array(params["_prev"], dtype=float)
        rising = abs(float(params.get("rising_slew", np.inf)))
        falling = abs(float(params.get("falling_slew", np.inf)))

        max_inc = rising * dt
        max_dec = falling * dt

        delta = u - prev
        delta = np.minimum(delta, max_inc)
        delta = np.maximum(delta, -max_dec)

        y = prev + delta
        params["_prev"] = y
        return {0: y}

