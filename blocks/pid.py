import numpy as np
from blocks.base_block import BaseBlock
from blocks.input_helpers import get_scalar


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
            "N": {
                "type": "float",
                "default": 20.0,
                "doc": "Derivative filter coefficient (higher = less smoothing).",
            },
            "u_min": {"type": "float", "default": -np.inf, "doc": "Output lower limit."},
            "u_max": {"type": "float", "default": np.inf, "doc": "Output upper limit."},
            "_init_start_": {"type": "bool", "default": True, "doc": "Internal init flag."},
            "sampling_time": {
                "type": "float",
                "default": -1.0,
                "doc": "Sample time (-1=continuous, 0=inherited, >0=discrete).",
            },
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

    def symbolic_execute(self, inputs, params):
        """
        Symbolic execution for equation extraction.

        PID transfer function: C(s) = Kp + Ki/s + Kd*N*s/(s + N)

        Args:
            inputs: Dict of symbolic input expressions {0: setpoint, 1: measurement}
            params: Dict of block parameters

        Returns:
            Dict of symbolic output expressions {0: C(s) * error}
        """
        try:
            from sympy import Symbol, simplify
        except ImportError:
            return None

        s = Symbol("s")

        # Get gains
        Kp = float(params.get("Kp", 1.0))
        Ki = float(params.get("Ki", 0.0))
        Kd = float(params.get("Kd", 0.0))
        N = float(params.get("N", 20.0))

        # Get input symbols (setpoint and measurement)
        sp = inputs.get(0, Symbol("sp"))
        meas = inputs.get(1, Symbol("meas"))
        e = sp - meas  # error = setpoint - measurement

        # PID transfer function: C(s) = Kp + Ki/s + Kd*N*s/(s + N)
        C_pid = Kp
        if Ki != 0:
            C_pid = C_pid + Ki / s
        if Kd != 0:
            C_pid = C_pid + Kd * N * s / (s + N)

        # Output = C(s) * error
        return {0: simplify(C_pid * e)}

    def get_symbolic_params(self, params):
        """Return symbolic parameters for equation extraction."""
        try:
            from sympy import Symbol

            return {
                "Kp": Symbol("K_p"),
                "Ki": Symbol("K_i"),
                "Kd": Symbol("K_d"),
            }
        except ImportError:
            return {}

    def execute(self, time, inputs, params, **kwargs):
        # Output-only path: when the caller asks for output only, or inputs are
        # missing (e.g. memory-block pre-population in Loop 1), return the last
        # computed output without mutating state. The PID's actual integration
        # happens in the normal Loop 2 execute call.
        #
        # The output_only flag must be honoured explicitly: PID is a memory
        # block, so the simulation loop's first pass calls it with
        # output_only=True while input_queue still holds the previous step's
        # inputs. Testing only for missing inputs let that pass integrate a
        # second time, doubling Ki's contribution on every timestep.
        # Port 1 (measurement) may legitimately be unconnected: that is the
        # error-input wiring, Sum(setpoint - measurement) -> PID, where port 0
        # already carries the error and the measurement is implicitly 0. The
        # compiled kernel has always supported it (build_pid leaves meas_src
        # None), so bailing here on a missing port 1 froze such a loop at zero
        # on the interpreted path only. Port 0 missing is still a genuine
        # un-fed call.
        if kwargs.get("output_only", False) or 0 not in inputs:
            return {0: np.atleast_1d(params.get("_last_output_", 0.0))}

        dt = max(float(params.get("dtime", 0.01)), 1e-12)
        sp = get_scalar(inputs, 0, 0.0)
        meas = get_scalar(inputs, 1, 0.0) if 1 in inputs else 0.0
        e = sp - meas

        first_call = bool(params.get("_init_start_", True))
        if first_call:
            params["_int"] = 0.0
            # The derivative filter state starts at zero, not at the first error
            # sample -- see the derivative branch below.
            params["_d_state"] = 0.0
            params["_init_start_"] = False

        Kp = float(params.get("Kp", 0.0))
        Ki = float(params.get("Ki", 0.0))
        Kd = float(params.get("Kd", 0.0))
        N = float(params.get("N", 20.0))

        # Integral update.  Nothing accrues on the first call: no time has
        # elapsed at t0, and integrating a whole step there makes the reported
        # integral lead the true one by one dt for the entire run (Ki=1 on a
        # unit error read 3.01 over a 3 s run).  This was masked for as long as
        # the simulation loop delivered the PID's output one step late, which
        # cancelled it exactly; see tests/regression/test_feedthrough_memory.py.
        if not first_call:
            params["_int"] += e * dt

        # Derivative with a first-order filter, stated exactly as the documented
        # C(s) = Kp + Ki/s + Kd*N*s/(s+N) and as the compiled kernel realises it
        # (lib/engine/compiler_kernels/state.py::build_pid): `_d_state` is the
        # low-passed *error*
        #
        #     d(x_d)/dt = N * (e - x_d),   x_d(0) = 0,   D term = Kd*N*(e - x_d),
        #
        # discretised with backward Euler, which is unconditionally stable so a
        # large N or a coarse dt cannot make the branch ring.
        #
        # It used to filter the finite difference (e[k] - e[k-1])/dt with
        # `_prev_e` seeded from the first error sample, which forces de = 0 at
        # t0 and throws away the derivative's entire response to a step in the
        # reference -- an O(1) error that never shrinks with dt.  Starting the
        # filter state at zero reproduces the continuous u(0+) = Kd*N*e(0).
        x_d = (params["_d_state"] + N * dt * e) / (1.0 + N * dt)
        params["_d_state"] = x_d
        d_term = Kd * N * (e - x_d)

        u = Kp * e + Ki * params["_int"] + d_term

        # Saturation and integral anti-windup (clamp integral within output bounds / Ki)
        u_min = params.get("u_min", -np.inf)
        u_max = params.get("u_max", np.inf)
        if u < u_min:
            u = u_min
            if Ki != 0:
                params["_int"] = (u_min - Kp * e - d_term) / Ki
        elif u > u_max:
            u = u_max
            if Ki != 0:
                params["_int"] = (u_max - Kp * e - d_term) / Ki

        params["_last_output_"] = (
            float(np.asarray(u).flatten()[0]) if hasattr(u, "flatten") else float(u)
        )
        return {0: np.atleast_1d(u)}
