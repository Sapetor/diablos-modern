import numpy as np
from blocks.base_block import BaseBlock


class SwitchBlock(BaseBlock):
    """
    Two-way selector: chooses between in1 and in2 based on control signal and threshold.
    Inputs: 0=control, 1=true path, 2=false path.
    """

    @property
    def block_name(self):
        return "Switch"

    @property
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return "Selects between two signals based on control >= threshold."

    @property
    def params(self):
        return {
            "threshold": {"type": "float", "default": 0.0, "doc": "Control threshold."},
        }

    @property
    def inputs(self):
        return [{"name": "ctrl", "type": "any"},
                {"name": "in_true", "type": "any"},
                {"name": "in_false", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        ctrl = float(np.atleast_1d(inputs[0])[0])
        sel = 1 if ctrl >= float(params.get("threshold", 0.0)) else 2
        out = inputs.get(sel)
        # Ensure ndarray for consistency
        if isinstance(out, (float, int)):
            out = np.atleast_1d(out)
        return {0: np.array(out, dtype=float)}
