"""
RateTransition block for multi-rate simulation.

Provides safe signal transfer between blocks running at different sample rates.
Handles both upsampling (slow→fast) and downsampling (fast→slow).
"""
from blocks.base_block import BaseBlock
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RateTransitionBlock(BaseBlock):
    """
    Rate Transition block for multi-rate simulation support.

    Handles signal conversion between different sample rates:
    - Upsampling (slow→fast): ZOH, Linear interpolation, or Filter
    - Downsampling (fast→slow): Sample, Average, or Filter

    The output_sample_time parameter defines the target output rate.
    """

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "RateTransition"

    @property
    def fn_name(self):
        return "rate_transition"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "output_sample_time": {"default": 0.1, "type": "float",
                                   "doc": "Target output sample period (seconds)"},
            "transition_mode": {"default": "ZOH", "type": "str",
                              "options": ["ZOH", "Linear", "Filter", "Sample", "Average"]},
            "filter_cutoff": {"default": 0.4, "type": "float"},  # Normalized cutoff for filter mode
            "sampling_time": {"default": -1.0, "type": "float",
                             "doc": "Block runs continuously (-1) for smooth output"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return (
            "Rate Transition Block"
            "\n\nSafely transfers signals between blocks running at different sample rates."
            "\n\nParameters:"
            "\n- Output Sample Time: Target output sample period (seconds). Set to -1 for continuous."
            "\n- Transition Mode: How to handle rate conversion:"
            "\n  - ZOH: Zero-order hold (hold last sample, good for upsampling)"
            "\n  - Linear: Linear interpolation between samples (ramps after each input change)"
            "\n  - Filter: Low-pass filter (anti-alias for downsampling)"
            "\n  - Sample: Take latest sample (simple downsampling)"
            "\n  - Average: Average samples in window (downsampling)"
            "\n- Filter Cutoff: Normalized cutoff frequency for Filter mode (0-0.5)"
            "\n\nUsage:"
            "\n- Place between blocks with different sample rates"
            "\n- For slow→fast: Use ZOH or Linear"
            "\n- For fast→slow: Use Filter, Sample, or Average"
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw rate transition icon (two different rates merging)."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw slow rate (wide steps) on left
        path.moveTo(0.1, 0.7)
        path.lineTo(0.3, 0.7)
        path.lineTo(0.3, 0.3)
        path.lineTo(0.5, 0.3)
        # Draw transition arrow
        path.moveTo(0.5, 0.5)
        path.lineTo(0.7, 0.5)
        # Draw fast rate (narrow steps) on right
        path.moveTo(0.7, 0.6)
        path.lineTo(0.8, 0.6)
        path.lineTo(0.8, 0.4)
        path.lineTo(0.9, 0.4)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """
        Rate transition: Convert signal between different sample rates.
        """
        output_only = kwargs.get('output_only', False)
        mode = str(params.get('transition_mode', 'ZOH'))


        # Initialize on first call
        if params.get('_init_start_', True):
            logger.info(f"RateTransition: Initializing at t={time}, mode={mode}")
            params['_init_start_'] = False
            params['_next_output_time_'] = 0.0
            params['_held_value_'] = 0.0
            params['_prev_value_'] = 0.0
            params['_prev_time_'] = 0.0
            params['_sample_buffer_'] = []
            params['_filter_state_'] = 0.0
            # Linear mode state
            params['_linear_start_val_'] = 0.0
            params['_linear_end_val_'] = 0.0
            params['_linear_start_time_'] = 0.0
            params['_linear_ramp_duration_'] = 0.1  # Default ramp duration
            params['_linear_last_input_'] = None
            # Initialize with input value if available
            val = inputs.get(0, 0.0)
            if hasattr(val, '__iter__') and not isinstance(val, str):
                val = np.atleast_1d(val)
            elif hasattr(val, 'item'):
                val = val.item()
            params['_held_value_'] = val
            params['_prev_value_'] = val
            params['_linear_start_val_'] = val
            params['_linear_end_val_'] = val
            params['_linear_last_input_'] = val

        output_sample_time = float(params.get('output_sample_time', 0.1))
        mode = str(params.get('transition_mode', 'ZOH'))
        filter_cutoff = float(params.get('filter_cutoff', 0.4))

        # Get current input
        val = inputs.get(0, params['_held_value_'])
        if hasattr(val, '__iter__') and not isinstance(val, str):
            val = np.atleast_1d(val)
        elif hasattr(val, 'item'):
            val = val.item()

        # Convert to float for comparison
        val_f = float(val) if not isinstance(val, np.ndarray) else float(np.atleast_1d(val).flat[0])


        # Continuous output mode (pass-through)
        if output_sample_time <= 0:
            return {0: val, 'E': False}

        # Linear mode: Handle ramp interpolation
        if mode == 'Linear' and not output_only:
            last_input = params['_linear_last_input_']
            if last_input is not None:
                last_f = float(last_input) if not isinstance(last_input, np.ndarray) else float(np.atleast_1d(last_input).flat[0])

                # Check if input changed (new sample from upstream)
                if abs(val_f - last_f) > 1e-9:
                    # Input changed! Start a new ramp
                    # Ramp from current output position to new input value
                    # Estimate ramp duration from time since last change
                    ramp_duration = time - params['_linear_start_time_']
                    if ramp_duration < 0.001:
                        ramp_duration = 0.1  # Default if too fast

                    logger.info(f"RateTransition Linear: input changed at t={time:.3f}, {last_f:.3f} -> {val_f:.3f}")
                    params['_linear_start_val_'] = params['_held_value_']  # Start from current output
                    params['_linear_end_val_'] = val_f
                    params['_linear_start_time_'] = time
                    params['_linear_ramp_duration_'] = ramp_duration
                    params['_linear_last_input_'] = val

        # Check if it's time to produce output
        at_output_time = time >= params['_next_output_time_'] - 1e-9

        if not output_only:
            # Store sample in buffer for averaging/filtering
            params['_sample_buffer_'].append((time, val))
            # Limit buffer size
            if len(params['_sample_buffer_']) > 100:
                params['_sample_buffer_'] = params['_sample_buffer_'][-50:]

        # Compute output based on mode
        if mode == 'Linear':
            # Linear interpolation: ramp from start_val to end_val
            start_val = params['_linear_start_val_']
            end_val = params['_linear_end_val_']
            start_time = params['_linear_start_time_']
            ramp_duration = params['_linear_ramp_duration_']

            if ramp_duration > 0 and time >= start_time:
                # Calculate interpolation factor
                alpha = (time - start_time) / ramp_duration
                alpha = min(1.0, max(0.0, alpha))
                output_val = start_val + alpha * (end_val - start_val)
            else:
                output_val = end_val


            if not output_only:
                params['_held_value_'] = output_val

            # Schedule next output time
            if at_output_time and not output_only:
                while params['_next_output_time_'] <= time + 1e-9:
                    params['_next_output_time_'] += output_sample_time

            return {0: output_val, 'E': False}

        # Non-Linear modes only update at output times
        if at_output_time:
            if not output_only:
                if mode == 'ZOH':
                    output_val = val

                elif mode == 'Sample':
                    output_val = val

                elif mode == 'Average':
                    buffer = params['_sample_buffer_']
                    if buffer:
                        vals = [v for t, v in buffer]
                        if isinstance(vals[0], np.ndarray):
                            output_val = np.mean(vals, axis=0)
                        else:
                            output_val = np.mean([float(v) for v in vals])
                        params['_sample_buffer_'] = []
                    else:
                        output_val = val

                elif mode == 'Filter':
                    alpha = min(1.0, max(0.01, 2.0 * np.pi * filter_cutoff * output_sample_time))
                    prev_filter = params['_filter_state_']
                    if isinstance(val, np.ndarray):
                        if not isinstance(prev_filter, np.ndarray):
                            prev_filter = np.zeros_like(val)
                        output_val = alpha * val + (1 - alpha) * prev_filter
                    else:
                        output_val = alpha * float(val) + (1 - alpha) * float(prev_filter)
                    params['_filter_state_'] = output_val

                else:
                    output_val = val

                params['_held_value_'] = output_val
                params['_prev_value_'] = val
                params['_prev_time_'] = time

                while params['_next_output_time_'] <= time + 1e-9:
                    params['_next_output_time_'] += output_sample_time

            return {0: params['_held_value_'], 'E': False}

        return {0: params['_held_value_'], 'E': False}
