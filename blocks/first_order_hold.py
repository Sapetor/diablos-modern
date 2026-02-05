"""
FirstOrderHold block for multi-rate simulation.

Provides linear interpolation between samples, smoother than ZOH
but introduces a one-sample delay.
"""
from blocks.base_block import BaseBlock
import numpy as np


class FirstOrderHoldBlock(BaseBlock):
    """
    First-Order Hold (FOH) block.

    Linearly interpolates between samples to produce a smoother output
    than Zero-Order Hold. Introduces a one-sample delay since it needs
    two samples to interpolate.
    """

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "FirstOrderHold"

    @property
    def fn_name(self):
        return "first_order_hold"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "input_sample_time": {"default": 0.1, "type": "float",
                                  "doc": "Sample period for input (seconds)"},
            "sampling_time": {"default": -1.0, "type": "float",
                             "doc": "Block runs continuously (-1) to interpolate"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return (
            "First-Order Hold (FOH)"
            "\n\nSamples input at fixed rate and linearly interpolates between samples."
            "\n\nParameters:"
            "\n- Sampling Time: The period (in seconds) between samples."
            "\n\nUsage:"
            "\n- Smoother output than Zero-Order Hold"
            "\n- Introduces one sample delay for interpolation"
            "\n- Good for continuous-to-discrete conversion when smoothness matters"
            "\n\nNote: Output = linear interpolation from previous sample to current sample"
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw FOH icon (ramp segments instead of stairs)."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw connected ramp segments (linear interpolation visualization)
        path.moveTo(0.1, 0.8)
        path.lineTo(0.35, 0.5)
        path.lineTo(0.6, 0.6)
        path.lineTo(0.9, 0.2)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """
        First-Order Hold: Samples input and linearly interpolates between samples.
        """
        output_only = kwargs.get('output_only', False)

        # Initialize on first call
        if params.get('_init_start_', True):
            params['_init_start_'] = False
            params['_sample_time_prev_'] = 0.0
            params['_sample_time_curr_'] = 0.0
            params['_next_sample_time_'] = 0.0
            # Initialize with input value
            val = inputs.get(0, 0.0)
            if hasattr(val, '__iter__') and not isinstance(val, str):
                val = np.atleast_1d(val)
            elif hasattr(val, 'item'):
                val = val.item()
            params['_value_prev_'] = val
            params['_value_curr_'] = val

        sampling_time = float(params.get('input_sample_time', params.get('sampling_time', 0.1)))
        if sampling_time < 0:
            sampling_time = 0.1  # Default if continuous marker was used

        # Get current input
        val = inputs.get(0, 0.0)
        if hasattr(val, '__iter__') and not isinstance(val, str):
            val = np.atleast_1d(val)
        elif hasattr(val, 'item'):
            val = val.item()

        # Check if it's time to take a new sample
        if time >= params['_next_sample_time_'] - 1e-9:
            if not output_only:
                # Shift samples: current becomes previous
                params['_value_prev_'] = params['_value_curr_']
                params['_sample_time_prev_'] = params['_sample_time_curr_']

                # Store new sample
                params['_value_curr_'] = val
                params['_sample_time_curr_'] = time

                # Schedule next sample
                while params['_next_sample_time_'] <= time + 1e-9:
                    params['_next_sample_time_'] += sampling_time

        # First-Order Hold: Extrapolate forward from the last two samples
        # This produces a ramp between sample times based on the computed slope
        t_prev = params['_sample_time_prev_']
        t_curr = params['_sample_time_curr_']
        v_prev = params['_value_prev_']
        v_curr = params['_value_curr_']

        # Compute slope and extrapolate from the most recent sample
        if t_curr > t_prev:
            # Compute slope between last two samples
            dt_samples = t_curr - t_prev
            # Time since the most recent sample
            dt_from_curr = time - t_curr

            if isinstance(v_curr, np.ndarray) and isinstance(v_prev, np.ndarray):
                slope = (v_curr - v_prev) / dt_samples
                output_val = v_curr + slope * dt_from_curr
            else:
                v_prev_f = float(v_prev) if not isinstance(v_prev, np.ndarray) else float(v_prev.flat[0])
                v_curr_f = float(v_curr) if not isinstance(v_curr, np.ndarray) else float(v_curr.flat[0])
                slope = (v_curr_f - v_prev_f) / dt_samples
                output_val = v_curr_f + slope * dt_from_curr
        else:
            # No slope yet (only one sample), just hold
            if isinstance(v_curr, np.ndarray):
                output_val = v_curr.copy()
            else:
                output_val = float(v_curr) if not isinstance(v_curr, np.ndarray) else float(v_curr.flat[0])

        return {0: output_val, 'E': False}
