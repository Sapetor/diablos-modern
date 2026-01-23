import numpy as np
from blocks.base_block import BaseBlock


class PIDBlock(BaseBlock):
    """
    PID controller with filtered derivative and anti-windup via integral clamping.
    Inputs: 0 = setpoint, 1 = measurement.
    Output: control signal.
    """

    @property
    def block_name(self):
        return "PID"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def doc(self):
        return (
            "PID Controller."
            "\n\nu(t) = P + I + D"
            "\n\nParameters:"
            "\n- Proportional (P): Kp * error"
            "\n- Integral (I): Ki * integral(error)"
            "\n- Derivative (D): Kd * derivative(error)"
            "\n- Filter Coeff (N): Derivative filter bandwidth (Low-pass)."
            "\n  D term = Kd * N * s / (s + N)"
            "\n\nUsage:"
            "\nFeedback control. Tuning parameters Kp, Ki, Kd."
        )

    @property
    def params(self):
        return {
            "Kp": {"type": "float", "default": 1.0, "doc": "Proportional gain."},
            "Ki": {"type": "float", "default": 0.0, "doc": "Integral gain."},
            "Kd": {"type": "float", "default": 0.0, "doc": "Derivative gain."},
            "N": {"type": "float", "default": 20.0, "doc": "Derivative filter coefficient (higher = less smoothing)."},
            "u_min": {"type": "float", "default": -np.inf, "doc": "Output lower limit."},
            "u_max": {"type": "float", "default": np.inf, "doc": "Output upper limit."},
            "_init_start_": {"type": "bool", "default": True, "doc": "Internal init flag."},
        }

    @property
    def inputs(self):
        return [{"name": "setpoint", "type": "any"}, {"name": "measurement", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "u", "type": "any"}]

    def draw_icon(self, block_rect):
        """PID uses text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params):
        dt = float(params.get("dtime", 0.01))
        sp = float(np.atleast_1d(inputs[0])[0])
        meas = float(np.atleast_1d(inputs[1])[0])
        e = sp - meas

        if params.get("_init_start_", True):
            params["_int"] = 0.0
            params["_d_state"] = 0.0
            params["_prev_e"] = e
            params["_init_start_"] = False

        Kp = float(params.get("Kp", 0.0))
        Ki = float(params.get("Ki", 0.0))
        Kd = float(params.get("Kd", 0.0))
        N = float(params.get("N", 20.0))

        # Integral update
        params["_int"] += e * dt

        # Derivative with first-order filter (bandwidth ~ N*Kd)
        de = (e - params["_prev_e"]) / dt
        alpha = N * dt / (1.0 + N * dt) if Kd != 0 else 0.0
        params["_d_state"] = params["_d_state"] + alpha * (de - params["_d_state"])
        params["_prev_e"] = e

        u = Kp * e + Ki * params["_int"] + Kd * params["_d_state"]

        # Saturation and integral anti-windup (clamp integral within output bounds / Ki)
        u_min = params.get("u_min", -np.inf)
        u_max = params.get("u_max", np.inf)
        if u < u_min:
            u = u_min
            if Ki != 0:
                params["_int"] = (u_min - Kp * e - Kd * params["_d_state"]) / Ki
        elif u > u_max:
            u = u_max
            if Ki != 0:
                params["_int"] = (u_max - Kp * e - Kd * params["_d_state"]) / Ki

        return {0: np.atleast_1d(u)}
