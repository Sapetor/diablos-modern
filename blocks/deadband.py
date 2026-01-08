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
        return "Dead zone: output is zero in [start, end], offset-subtracted outside."

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

