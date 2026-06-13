import bisect
import numpy as np
from collections import deque
from blocks.base_block import BaseBlock


class VariableTransportDelayBlock(BaseBlock):
    """
    Variable (input-driven) transport delay.

    Like a fixed TransportDelay, but the delay time τ is supplied at runtime
    on a second input port rather than being a fixed parameter:

        y(t) = u(t - τ(t))

    where τ is read from input port 1 and clamped to [0, max_delay]. Uses
    linear interpolation of a (time, value) history buffer for sub-sample
    accuracy.
    """

    @property
    def block_name(self):
        return "VariableTransportDelay"

    @property
    def fn_name(self):
        return "variable_transport_delay"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "cyan"

    @property
    def b_type(self):
        """Memory block - holds past samples; can break algebraic loops."""
        return 1

    @property
    def doc(self):
        return (
            "Variable Transport Delay / Input-Driven Time Delay."
            "\n\nDelays the signal input by a delay τ supplied on a second port."
            "\ny(t) = u(t - τ(t))"
            "\n\nInputs:"
            "\n- Port 0: signal to be delayed."
            "\n- Port 1: delay τ in seconds (clamped to [0, Max Delay])."
            "\n\nParameters:"
            "\n- Max Delay: Upper bound on τ and buffer retention window."
            "\n- Initial Value: Output before the requested sample exists."
            "\n\nUsage:"
            "\nModels variable latency: pipe/conveyor flow at changing speeds,"
            "\nnetwork jitter, or any time-varying transport lag."
        )

    @property
    def params(self):
        return {
            "max_delay": {"type": "float", "default": 1.0, "doc": "Maximum delay τ (s); also the buffer retention window."},
            "initial_value": {"type": "float", "default": 0.0, "doc": "Output before the requested sample exists."},
            "_time_buffer_": {"type": "list", "default": [], "doc": "Internal time history (do not edit)."},
            "_value_buffer_": {"type": "list", "default": [], "doc": "Internal value history (do not edit)."},
            "_init_start_": {"type": "bool", "default": True, "doc": "Initialization flag."},
        }

    @property
    def inputs(self):
        return [
            {"name": "in", "type": "any"},
            {"name": "tau", "type": "any"},
        ]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        max_delay = max(0.0, float(params.get("max_delay", 1.0)))
        initial_value = float(params.get("initial_value", 0.0))

        # Initialize buffers on first call, seeding with the initial value so
        # that requests at/before t0 have a defined value to interpolate from.
        if params.get("_init_start_", True):
            params["_time_buffer_"] = deque()
            params["_value_buffer_"] = deque()
            params["_time_buffer_"].append(float(time))
            params["_value_buffer_"].append(np.atleast_1d(initial_value))
            params["_init_start_"] = False

        time_buffer = params["_time_buffer_"]
        value_buffer = params["_value_buffer_"]

        # Read the requested delay from port 1 (default 0 if not connected),
        # then clamp to the implicit [0, max_delay] window.
        tau_raw = inputs.get(1, 0.0)
        tau = float(np.atleast_1d(tau_raw)[0])
        tau = min(max(tau, 0.0), max_delay)
        target_time = float(time) - tau

        # Output-only path (init / no signal input): hold without recording a
        # spurious sample into the interpolation history.
        if kwargs.get("output_only") or 0 not in inputs:
            return {0: self._interpolate(time_buffer, value_buffer, target_time, initial_value)}

        # Record the current (time, signal) sample.
        current_input = np.atleast_1d(inputs.get(0, initial_value))
        time_buffer.append(float(time))
        value_buffer.append(current_input.copy())

        output = self._interpolate(time_buffer, value_buffer, target_time, initial_value)

        # Prune entries older than the full max_delay window. Using max_delay
        # (rather than a hardcoded multiple of the *current* tau) guarantees a
        # large random tau never reads into a pruned/stale region.
        prune_time = float(time) - max_delay
        while len(time_buffer) > 2 and time_buffer[1] <= prune_time:
            time_buffer.popleft()
            value_buffer.popleft()

        params["_time_buffer_"] = time_buffer
        params["_value_buffer_"] = value_buffer

        return {0: output}

    def _interpolate(self, time_buffer, value_buffer, target_time, initial_value):
        """Linear interpolation of the buffer at target_time."""
        if len(time_buffer) == 0:
            return np.atleast_1d(initial_value)

        # Request at/after the most recent sample (tau == 0 -> passthrough).
        # Checked before the earliest-sample branch so passthrough wins at the
        # seeded buffer start (where buffer[0] and buffer[-1] share a timestamp).
        if target_time >= time_buffer[-1]:
            return value_buffer[-1].copy()

        # Request precedes the start of recorded history.
        if target_time <= time_buffer[0]:
            return np.atleast_1d(initial_value)

        # Find bracketing indices via binary search (O(log n)).
        i = bisect.bisect_right(time_buffer, target_time) - 1
        i = max(0, min(i, len(time_buffer) - 2))

        t0 = time_buffer[i]
        t1 = time_buffer[i + 1]

        if t1 - t0 == 0:
            return value_buffer[i].copy()

        alpha = (target_time - t0) / (t1 - t0)
        v0 = value_buffer[i]
        v1 = value_buffer[i + 1]
        return (1.0 - alpha) * v0 + alpha * v1
