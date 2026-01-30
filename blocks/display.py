import numpy as np
from blocks.base_block import BaseBlock


class DisplayBlock(BaseBlock):
    """
    Displays the current input value numerically on the canvas.
    Useful for monitoring signal values during simulation.
    """

    @property
    def block_name(self):
        return "Display"

    @property
    def category(self):
        return "Sinks"

    @property
    def color(self):
        return "red"

    @property
    def doc(self):
        return (
            "Numerical Display."
            "\n\nShows the current value of the input signal."
            "\n\nParameters:"
            "\n- Format: standard Python f-string format (e.g. {:.2f})."
            "\n- Label: Text label prefix."
            "\n\nUsage:"
            "\nMonitor scalar values during simulation."
        )

    @property
    def params(self):
        return {
            "format": {"type": "string", "default": "%.3f", "doc": "Printf-style format string."},
            "label": {"type": "string", "default": "", "doc": "Optional label prefix."},
            "_display_value_": {"type": "string", "default": "---", "doc": "Current displayed value (internal)."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    @property
    def requires_outputs(self):
        """Sinks don't need output connections."""
        return False

    def draw_icon(self, block_rect):
        """Display uses dynamic text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        fmt = params.get("format", "%.3f")
        label = params.get("label", "")
        
        input_value = inputs.get(0, 0)
        
        try:
            # Handle arrays - show first element or array summary
            if hasattr(input_value, '__len__') and len(input_value) > 1:
                # For arrays, show first few values
                vals = np.atleast_1d(input_value).flatten()
                if len(vals) <= 3:
                    formatted = "[" + ", ".join([fmt % v for v in vals]) + "]"
                else:
                    formatted = "[" + ", ".join([fmt % v for v in vals[:3]]) + ", ...]"
            else:
                # Single value
                val = float(np.atleast_1d(input_value)[0])
                formatted = fmt % val
        except (ValueError, TypeError):
            formatted = str(input_value)
        
        # Add label prefix if specified
        if label:
            params["_display_value_"] = f"{label}: {formatted}"
        else:
            params["_display_value_"] = formatted
        
        # Display blocks are pass-through (no output)
        return {0: np.atleast_1d(input_value)}
