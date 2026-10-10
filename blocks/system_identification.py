"""System Identification source block.

Fits a continuous transfer function ``G(s) = B(s)/A(s)`` of the chosen orders
to a recorded input/output experiment (CSV / NPZ / MAT / text, read with the
same loader as FromFile), then replays the fitted model's response to the
recorded input next to the measured output, so a Scope shows how good the fit
is. The identified coefficients and fit percentage are logged and kept in
``params`` (``_fit_num_``, ``_fit_den_``, ``_fit_percent_``) to be copied into
a Transfer Function block.

Both outputs are functions of time only (interpolated from arrays computed
once at initialization), so the block carries no integration state.
"""

import logging

import numpy as np

from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class SystemIdentificationBlock(BaseBlock):
    """Identify an n-th order transfer function from file data."""

    @property
    def block_name(self):
        return "SystemIdentification"

    @property
    def display_name(self):
        return "System Identification"

    @property
    def category(self):
        return "Sources"

    @property
    def b_type(self):
        return 0

    @property
    def color(self):
        return "mediumseagreen"

    @property
    def requires_inputs(self):
        return False

    @property
    def doc(self):
        return (
            "Fits a transfer function G(s) = B(s)/A(s) to a recorded input/output "
            "experiment and replays the fitted model next to the measurement.\n\n"
            "Parameters:\n"
            "- data_file: path to the data file (.csv, .npz, .mat, .txt).\n"
            "- time_col, input_col, output_col: column names (CSV header / NPZ-MAT "
            "key) or 0-based numeric indices.\n"
            "- num_order: order of B(s) (0 = no zeros), at most den_order.\n"
            "- den_order: order of A(s), the number of poles.\n\n"
            "Outputs:\n"
            "- model: the fitted G(s) driven by the recorded input.\n"
            "- measured: the recorded output.\n\n"
            "The fitted numerator/denominator and the fit percentage (100 = "
            "perfect, NRMSE as in MATLAB's compare) are written to the log when "
            "the run starts; copy them into a Transfer Function block to use the "
            "model in a design."
        )

    @property
    def params(self):
        return {
            "data_file": {
                "type": "string",
                "default": "",
                "doc": "Path to the data file (.csv/.npz/.mat/.txt).",
            },
            "time_col": {
                "type": "string",
                "default": "t",
                "doc": "Time column: name or 0-based numeric index.",
            },
            "input_col": {
                "type": "string",
                "default": "u",
                "doc": "Input (excitation) column: name or 0-based numeric index.",
            },
            "output_col": {
                "type": "string",
                "default": "y",
                "doc": "Output (response) column: name or 0-based numeric index.",
            },
            "num_order": {
                "type": "int",
                "default": 0,
                "doc": "Numerator order (number of zeros), at most the denominator order.",
            },
            "den_order": {
                "type": "int",
                "default": 2,
                "doc": "Denominator order (number of poles).",
            },
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "model", "type": "float"}, {"name": "measured", "type": "float"}]

    def draw_icon(self, block_rect):
        """Measured points with a fitted step response through them."""
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        for x, y in ((0.22, 0.70), (0.34, 0.45), (0.48, 0.33), (0.64, 0.29), (0.80, 0.30)):
            path.addEllipse(x - 0.035, y - 0.035, 0.07, 0.07)
        path.moveTo(0.15, 0.82)
        path.cubicTo(0.30, 0.35, 0.50, 0.28, 0.88, 0.29)
        return path

    def _identify(self, params):
        """Load the file and fit the model once per run. Returns an error or None."""
        key = (
            str(params.get("data_file", "") or ""),
            str(params.get("time_col", "t")),
            str(params.get("input_col", "u")),
            str(params.get("output_col", "y")),
            int(params.get("num_order", 0)),
            int(params.get("den_order", 2)),
        )
        if not params.get("_init_start_", True) and params.get("_fit_key_") == key:
            return params.get("_fit_error_")
        params["_init_start_"] = False
        params["_fit_key_"] = key
        params["_fit_error_"] = None
        for stale in ("_grid_", "_y_model_", "_t_data_", "_y_data_"):
            params.pop(stale, None)

        data_file, time_col, input_col, output_col, num_order, den_order = key
        if not data_file:
            params["_fit_error_"] = "No data file specified"
            return params["_fit_error_"]

        from lib.analysis.sysid import SysIdError, format_tf, identify_tf
        from lib.services.timeseries_loader import TimeseriesLoadError, load_timeseries

        try:
            t_u, u = load_timeseries(data_file, time_col, input_col)
            t_y, y = load_timeseries(data_file, time_col, output_col)
            result = identify_tf(t_u, u, np.interp(t_u, t_y, y), num_order, den_order)
        except (TimeseriesLoadError, SysIdError) as exc:
            params["_fit_error_"] = f"System identification: {exc}"
            logger.error(params["_fit_error_"])
            return params["_fit_error_"]

        order = np.argsort(t_y, kind="stable")
        params["_t_data_"], params["_y_data_"] = t_y[order], y[order]
        params["_grid_"], params["_y_model_"] = result["grid"], result["y_model"]
        params["_fit_num_"], params["_fit_den_"] = result["num"], result["den"]
        params["_fit_percent_"] = result["fit_percent"]
        logger.info(
            "%s: G(s) = %s, fit %.1f %% (num=%s, den=%s)",
            params.get("_name_", "SystemIdentification"),
            format_tf(result["num"], result["den"]),
            result["fit_percent"],
            [round(v, 6) for v in result["num"]],
            [round(v, 6) for v in result["den"]],
        )
        return None

    def execute(self, time, inputs, params, **kwargs):
        error = self._identify(params)
        if error:
            return {"E": True, "error": error}
        t = float(time)
        model = float(np.interp(t, params["_grid_"], params["_y_model_"]))
        measured = float(np.interp(t, params["_t_data_"], params["_y_data_"]))
        return {0: np.atleast_1d(model), 1: np.atleast_1d(measured), "E": False}
