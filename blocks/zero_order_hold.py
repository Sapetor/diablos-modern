from blocks.base_block import BaseBlock


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
        output_only = kwargs.get('output_only', False)
        
        if params.get('_init_start_', True):
            params['_init_start_'] = False
            params['_next_sample_time_'] = 0.0
            # Initialize held value with initial input if available, else 0
            val = inputs.get(0, 0.0)
            if hasattr(val, 'item'):
                val = val.item()
            params['_held_value_'] = float(val)

        # Get current held value
        held_val = params['_held_value_']
        
        # Check if it's time to sample
        sampling_time = params['sampling_time']
        if time >= params['_next_sample_time_'] - 1e-9:
            if not output_only:
                # Update held value
                val = inputs.get(0, 0.0)
                if hasattr(val, 'item'):
                    val = val.item()
                params['_held_value_'] = float(val)
                
                # Schedule next sample
                while params['_next_sample_time_'] <= time + 1e-9:
                    params['_next_sample_time_'] += sampling_time
            
            return {0: params['_held_value_'], 'E': False}
            
        return {0: held_val, 'E': False}
