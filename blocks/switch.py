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
        return "Selects between inputs based on control. Modes: threshold (binary) or index (multi-way)."

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
        # Place control separate (index 0), then data inputs
        inputs = [{"name": "ctrl", "type": "any", "group": "control"}]
        for i in range(n):
            inputs.append({"name": f"in{i}", "type": "any"})
        return inputs

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

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
