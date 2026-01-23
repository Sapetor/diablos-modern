import numpy as np
from blocks.base_block import BaseBlock


class SwitchBlock(BaseBlock):
    """
    Two-way selector: chooses between in1 and in2 based on control signal and threshold.
    Inputs: 0=control, data inputs follow.
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
        return (
            "Signal Switch."
            "\n\nPasses one of the inputs based on the Control signal (middle port)."
            "\n\nCriteria:"
            "\n- u2 >= Threshold: Output = u1 (Top port)"
            "\n- u2 < Threshold:  Output = u3 (Bottom port)"
            "\n\nParameters:"
            "\n- Threshold: Switching value."
            "\n\nUsage:"
            "\nConditional logic or selecting between valid signals."
        )

    @property
    def params(self):
        return {
            "threshold": {"type": "float", "default": 0.0, "doc": "Control threshold (threshold mode)."},
            "n_inputs": {"type": "int", "default": 2, "doc": "Number of data inputs (>=2)."},
            "mode": {"type": "string", "default": "threshold", "doc": "'threshold' or 'index'."},
        }

    @property
    def inputs(self):
        return self.get_inputs()

    def get_inputs(self, params=None):
        p = params or {}
        raw_n = p.get("n_inputs", 2)
        if isinstance(raw_n, dict):
            raw_n = raw_n.get("default", 2)
        n = max(2, int(raw_n))
        # Place control on top, then data inputs on left
        inputs = [{"name": "ctrl", "type": "any", "position": "top"}]
        for i in range(n):
            inputs.append({"name": f"in{i}", "type": "any"})
        return inputs

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw switch selector icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Control arrow from top
        path.moveTo(0.5, 0.10)
        path.lineTo(0.5, 0.35)
        path.moveTo(0.47, 0.30); path.lineTo(0.5, 0.35); path.lineTo(0.53, 0.30)
        # Selector box
        path.moveTo(0.30, 0.35)
        path.lineTo(0.70, 0.35)
        path.lineTo(0.70, 0.75)
        path.lineTo(0.30, 0.75)
        path.lineTo(0.30, 0.35)
        # Data inputs  
        path.moveTo(0.30, 0.45); path.lineTo(0.45, 0.45)
        path.moveTo(0.30, 0.65); path.lineTo(0.45, 0.65)
        # Selected path
        path.moveTo(0.45, 0.45)
        path.lineTo(0.55, 0.55)
        path.lineTo(0.70, 0.55)
        # Output
        path.moveTo(0.70, 0.55); path.lineTo(0.90, 0.55)
        return path

    def execute(self, time, inputs, params):
        ctrl = float(np.atleast_1d(inputs[0])[0])
        mode = params.get("mode", "threshold")
        n = max(2, int(params.get("n_inputs", 2)))

        if mode == "index":
            sel = int(round(ctrl))
        else:
            sel = 0 if ctrl >= float(params.get("threshold", 0.0)) else 1
        sel = max(0, min(n - 1, sel))

        out = inputs.get(sel + 1)  # data inputs start at index 1
        if isinstance(out, (float, int)):
            out = np.atleast_1d(out)
        return {0: np.array(out, dtype=float)}

