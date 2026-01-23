import numpy as np
from blocks.base_block import BaseBlock


class TransportDelayBlock(BaseBlock):
    """
    Continuous-time transport delay: e^(-τs)
    Delays the input signal by a specified time duration (in seconds).
    Uses linear interpolation for sub-sample accuracy.
    """

    @property
    def block_name(self):
        return "TransportDelay"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "cyan"

    @property
    def doc(self):
        return (
            "Transport Delay / Time Delay."
            "\n\nDelays the input signal by a specified time amount."
            "\ny(t) = u(t - Delay)"
            "\n\nParameters:"
            "\n- Time Delay: Amount of delay in seconds."
            "\n- Initial Output: Output value before t < Delay."
            "\n- Buffer Size: Max history length (increase if simulation is long/fast)."
            "\n\nUsage:"
            "\nModels pipe flow, conveyor belts, or communication latency."
        )

    @property
    def params(self):
        return {
            "delay_time": {"type": "float", "default": 0.1, "doc": "Delay time τ in seconds."},
            "initial_value": {"type": "float", "default": 0.0, "doc": "Output before delay time elapses."},
            "_time_buffer_": {"type": "list", "default": [], "doc": "Internal time history (do not edit)."},
            "_value_buffer_": {"type": "list", "default": [], "doc": "Internal value history (do not edit)."},
            "_init_start_": {"type": "bool", "default": True, "doc": "Initialization flag."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        delay_time = max(0.0, float(params.get("delay_time", 0.1)))
        initial_value = float(params.get("initial_value", 0.0))
        
        # Initialize buffers on first call
        if params.get("_init_start_", True):
            params["_time_buffer_"] = []
            params["_value_buffer_"] = []
            params["_init_start_"] = False
        
        time_buffer = params["_time_buffer_"]
        value_buffer = params["_value_buffer_"]
        
        # Get current input
        current_input = np.atleast_1d(inputs.get(0, initial_value))
        
        # Add current time and value to buffer
        time_buffer.append(float(time))
        value_buffer.append(current_input.copy())
        
        # Target time for output (current time minus delay)
        target_time = float(time) - delay_time
        
        # If target time is before simulation start, return initial value
        if target_time < 0 or len(time_buffer) < 2:
            output = np.atleast_1d(initial_value)
        else:
            # Find interpolation points
            output = self._interpolate(time_buffer, value_buffer, target_time, initial_value)
        
        # Prune old buffer entries (keep only what's needed for delay)
        # Keep a margin of 2x delay time for safety
        prune_time = float(time) - 2.0 * delay_time
        while len(time_buffer) > 2 and time_buffer[0] < prune_time:
            time_buffer.pop(0)
            value_buffer.pop(0)
        
        params["_time_buffer_"] = time_buffer
        params["_value_buffer_"] = value_buffer
        
        return {0: output}
    
    def _interpolate(self, time_buffer, value_buffer, target_time, initial_value):
        """Linear interpolation to get value at target_time."""
        # If target time is before first recorded sample
        if target_time <= time_buffer[0]:
            return np.atleast_1d(initial_value)
        
        # If target time is after last recorded sample (shouldn't happen normally)
        if target_time >= time_buffer[-1]:
            return value_buffer[-1].copy()
        
        # Find bracketing indices
        for i in range(len(time_buffer) - 1):
            t0 = time_buffer[i]
            t1 = time_buffer[i + 1]
            
            if t0 <= target_time <= t1:
                # Linear interpolation
                if t1 - t0 == 0:
                    return value_buffer[i].copy()
                
                alpha = (target_time - t0) / (t1 - t0)
                v0 = value_buffer[i]
                v1 = value_buffer[i + 1]
                return (1.0 - alpha) * v0 + alpha * v1
        
        # Fallback (shouldn't reach here)
        return value_buffer[-1].copy()
