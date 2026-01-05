import numpy as np
from blocks.base_block import BaseBlock


class XYGraphBlock(BaseBlock):
    """
    Plots the first input (X) against the second input (Y).
    Creates a parametric plot instead of time-based plot.
    """

    @property
    def block_name(self):
        return "XYGraph"

    @property
    def category(self):
        return "Sinks"

    @property
    def color(self):
        return "red"

    @property
    def doc(self):
        return "Plots input X vs input Y. Useful for phase portraits and parametric curves."

    @property
    def params(self):
        return {
            "x_label": {"type": "string", "default": "X", "doc": "X-axis label."},
            "y_label": {"type": "string", "default": "Y", "doc": "Y-axis label."},
            "title": {"type": "string", "default": "XY Plot", "doc": "Plot title."},
            "_x_data_": {"type": "list", "default": [], "doc": "Stored X values (internal)."},
            "_y_data_": {"type": "list", "default": [], "doc": "Stored Y values (internal)."},
            "_init_start_": {"type": "bool", "default": True, "doc": "Initialization flag."},
        }

    @property
    def inputs(self):
        return [
            {"name": "x", "type": "any"},
            {"name": "y", "type": "any"}
        ]

    @property
    def outputs(self):
        return []

    @property
    def requires_outputs(self):
        """Sinks don't need output connections."""
        return False

    def execute(self, time, inputs, params):
        # Initialize on first call
        if params.get("_init_start_", True):
            params["_x_data_"] = []
            params["_y_data_"] = []
            params["_init_start_"] = False
        
        # Get input values
        x_val = float(np.atleast_1d(inputs.get(0, 0))[0])
        y_val = float(np.atleast_1d(inputs.get(1, 0))[0])
        
        # Store data points
        params["_x_data_"].append(x_val)
        params["_y_data_"].append(y_val)
        
        # No output needed for sink
        return {0: np.array([0.0])}
