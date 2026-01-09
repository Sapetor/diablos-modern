import numpy as np
from blocks.base_block import BaseBlock


class HysteresisBlock(BaseBlock):
    """
    Relay with hysteresis: switches output high/low with upper/lower thresholds.
    """

    @property
    def block_name(self):
        return "Hysteresis"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def doc(self):
        return "Relay with upper/lower thresholds; holds state between."

    @property
    def params(self):
        return {
            "upper": {"type": "float", "default": 0.5, "doc": "Threshold to switch high."},
            "lower": {"type": "float", "default": -0.5, "doc": "Threshold to switch low."},
            "high": {"type": "float", "default": 1.0, "doc": "Output when high."},
            "low": {"type": "float", "default": 0.0, "doc": "Output when low."},
            "_init_start_": {"type": "bool", "default": True, "doc": "Internal init flag."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw hysteresis loop icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Low -> High path
        path.moveTo(0.15, 0.75)
        path.lineTo(0.75, 0.75)
        path.lineTo(0.75, 0.25)
        path.lineTo(0.85, 0.25)
        # High -> Low path
        path.moveTo(0.85, 0.25)
        path.lineTo(0.25, 0.25)
        path.lineTo(0.25, 0.75)
        path.lineTo(0.15, 0.75)
        return path

    def execute(self, time, inputs, params):
        u = float(np.atleast_1d(inputs[0])[0])

        if params.get("_init_start_", True):
            # Initialize state based on input
            if u >= params["upper"]:
                params["_state"] = float(params["high"])
            elif u <= params["lower"]:
                params["_state"] = float(params["low"])
            else:
                params["_state"] = float(params["low"])
            params["_init_start_"] = False

        if u >= params["upper"]:
            params["_state"] = float(params["high"])
        elif u <= params["lower"]:
            params["_state"] = float(params["low"])

        return {0: np.atleast_1d(params["_state"])}

