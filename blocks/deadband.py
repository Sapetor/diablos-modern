import numpy as np
from blocks.base_block import BaseBlock


class DeadbandBlock(BaseBlock):
    """
    Dead Zone block matching Simulink behavior.
    
    Output is zero inside [start, end] deadzone.
    Outside the deadzone, the offset is subtracted:
    - input < start: output = input - start
    - start <= input <= end: output = 0
    - input > end: output = input - end
    """

    @property
    def block_name(self):
        return "Deadband"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def doc(self):
        return (
            "Dead Zone / Deadband."
            "\n\nOutputs zero when the input is within the specified range [Start, End]."
            "\n\nFunction:"
            "\n- u < Start: y = u - Start"
            "\n- Start <= u <= End: y = 0"
            "\n- u > End: y = u - End"
            "\n\nParameters:"
            "\n- Start/End: Lower and Upper bounds of the zero region."
            "\n\nUsage:"
            "\nModels mechanical play (backlash) or noise thresholds."
        )

    @property
    def params(self):
        return {
            "start": {"type": "float", "default": -0.5, "doc": "Start of dead zone (lower threshold)."},
            "end": {"type": "float", "default": 0.5, "doc": "End of dead zone (upper threshold)."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw deadband characteristic icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Left slope
        path.moveTo(0.15, 0.80)
        path.lineTo(0.35, 0.50)
        # Deadband zone (flat)
        path.lineTo(0.65, 0.50)
        # Right slope
        path.lineTo(0.85, 0.20)
        # X-axis reference
        path.moveTo(0.2, 0.5)
        path.lineTo(0.8, 0.5)
        return path

    def execute(self, time, inputs, params):
        u = np.array(inputs[0], dtype=float)
        start = float(params.get("start", -0.5))
        end = float(params.get("end", 0.5))
        
        # Simulink Dead Zone behavior
        out = np.zeros_like(u)
        
        # Below dead zone: subtract start threshold
        below_mask = u < start
        out[below_mask] = u[below_mask] - start
        
        # Above dead zone: subtract end threshold
        above_mask = u > end
        out[above_mask] = u[above_mask] - end
        
        # Inside dead zone: output remains 0 (already initialized)
        
        return {0: out}

