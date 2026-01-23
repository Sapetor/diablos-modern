import numpy as np
from blocks.base_block import BaseBlock


class DelayBlock(BaseBlock):
    """
    Delays the input signal by a specified number of time steps.
    Uses a circular buffer to store past values.
    """

    @property
    def block_name(self):
        return "Delay"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "cyan"

    @property
    def doc(self):
        return (
            "Discrete Integer Delay (z^-N)."
            "\n\nDelays the input by a fixed number of execution steps."
            "\ny[k] = u[k - N]"
            "\n\nParameters:"
            "\n- Delay Steps: Number of steps (N)."
            "\n- Initial Value: Output for k < N."
            "\n\nUsage:"
            "\nModels digital latency or buffer pipelines."
        )

    @property
    def params(self):
        return {
            "delay_steps": {"type": "int", "default": 1, "doc": "Number of time steps to delay."},
            "initial_value": {"type": "float", "default": 0.0, "doc": "Output before delay buffer fills."},
            "_buffer_": {"type": "list", "default": [], "doc": "Internal buffer (do not edit)."},
            "_init_start_": {"type": "bool", "default": True, "doc": "Initialization flag."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    @property
    def b_type(self):
        """Delay is a memory block - can break algebraic loops."""
        return 1

    def execute(self, time, inputs, params, output_only=False, **kwargs):
        delay_steps = max(1, int(params.get("delay_steps", 1)))
        initial_value = params.get("initial_value", 0.0)
        
        # Initialize buffer on first call
        if params.get("_init_start_", True):
            params["_buffer_"] = [np.atleast_1d(initial_value)] * delay_steps
            params["_init_start_"] = False
        
        buffer = params["_buffer_"]
        
        # For output_only mode (used in init), just return current buffer head
        if output_only:
            return {0: buffer[0] if buffer else np.atleast_1d(initial_value)}
        
        current_input = np.atleast_1d(inputs.get(0, initial_value))
        
        # Get oldest value (output) and add new input
        output = buffer[0]
        buffer.pop(0)
        buffer.append(current_input)
        params["_buffer_"] = buffer
        
        return {0: output}
