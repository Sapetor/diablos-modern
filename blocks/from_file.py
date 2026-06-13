"""FromFile / Data Import source block.

Replays a recorded time-series (CSV / NPZ / MAT / whitespace-text) as a driving
signal, interpolating the file's samples onto the simulation time grid. This is
the canonical way to feed real, recorded data (experimental logs, measured
references) into a diagram for model-vs-experiment overlay and data-driven
parameter identification.

The file is parsed once (via the shared :func:`lib.services.timeseries_loader.
load_timeseries`) and the resulting arrays are cached in ``params`` under
underscore-prefixed keys, per the engine's state-in-params rule. The cache is
rebuilt whenever the simulation re-initializes (``_init_start_``) or the file
path changes, so editing the path in the property panel takes effect on the next
run.
"""

import logging

import numpy as np

from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class FromFileBlock(BaseBlock):
    """Source block that replays a time-series loaded from a file."""

    @property
    def block_name(self):
        return "FromFile"

    @property
    def category(self):
        return "Sources"

    @property
    def b_type(self):
        """Source block - generates output without requiring input."""
        return 0

    @property
    def color(self):
        return "mediumseagreen"

    @property
    def requires_inputs(self):
        """Source block: no inputs need to be connected."""
        return False

    @property
    def doc(self):
        return (
            "Replays a recorded time-series from a file as a driving signal.\n\n"
            "Reads (time, signal) columns from a CSV / NPZ / MAT / text file and "
            "outputs the value interpolated to the current simulation time.\n\n"
            "Parameters:\n"
            "- data_file: path to the data file (.csv, .npz, .mat, .txt).\n"
            "- time_col: time column — a name (CSV header / NPZ-MAT key) or a "
            "0-based numeric index.\n"
            "- signal_col: signal column — name or numeric index.\n"
            "- interpolation: 'linear', 'zoh' (zero-order hold / step), or "
            "'nearest'.\n"
            "- end_behavior: 'hold' (clamp to the last sample) or 'loop' "
            "(restart from the beginning) once sim time passes the last sample.\n\n"
            "Usage:\nDrive a model with experimental data for model-vs-experiment "
            "overlay or system identification. Runs on the interpreter path."
        )

    @property
    def params(self):
        return {
            "data_file": {"type": "string", "default": "",
                          "doc": "Path to the data file (.csv/.npz/.mat/.txt)."},
            "time_col": {"type": "string", "default": "t",
                         "doc": "Time column: name or 0-based numeric index."},
            "signal_col": {"type": "string", "default": "y",
                           "doc": "Signal column: name or 0-based numeric index."},
            "interpolation": {
                "type": "choice",
                "default": "linear",
                "options": ["linear", "zoh", "nearest"],
                "doc": "Interpolation: linear, zoh (step), or nearest.",
            },
            "end_behavior": {
                "type": "choice",
                "default": "hold",
                "options": ["hold", "loop"],
                "doc": "Past the last sample: hold the last value or loop.",
            },
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "float"}]

    def draw_icon(self, block_rect):
        """Draw a document/data-file glyph in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Page outline with a folded corner.
        path.moveTo(0.30, 0.15)
        path.lineTo(0.62, 0.15)
        path.lineTo(0.72, 0.28)
        path.lineTo(0.72, 0.85)
        path.lineTo(0.30, 0.85)
        path.closeSubpath()
        # A couple of "text" lines.
        path.moveTo(0.37, 0.40)
        path.lineTo(0.65, 0.40)
        path.moveTo(0.37, 0.55)
        path.lineTo(0.65, 0.55)
        path.moveTo(0.37, 0.70)
        path.lineTo(0.55, 0.70)
        return path

    # ------------------------------------------------------------------ helpers
    def _ensure_loaded(self, params):
        """Load (and cache) the file data if needed. Returns True on success.

        Rebuilds the cache when the run re-initializes (``_init_start_``) or the
        configured path differs from the cached one. Stores ``_t_data_`` /
        ``_y_data_`` (sorted by time) and ``_load_error_`` in ``params``.
        """
        data_file = str(params.get("data_file", "") or "")
        needs_load = (
            params.get("_init_start_", True)
            or "_t_data_" not in params
            or params.get("_loaded_file_") != data_file
        )
        if not needs_load:
            return params.get("_load_error_") is None

        params["_init_start_"] = False
        params["_loaded_file_"] = data_file
        params["_load_error_"] = None

        if not data_file:
            params["_load_error_"] = "No data file specified"
            params.pop("_t_data_", None)
            params.pop("_y_data_", None)
            return False

        from lib.services.timeseries_loader import (
            load_timeseries,
            TimeseriesLoadError,
        )
        try:
            t_data, y_data = load_timeseries(
                data_file,
                params.get("time_col", "t"),
                params.get("signal_col", "y"),
            )
        except TimeseriesLoadError as exc:
            logger.error("FromFile: %s", exc)
            params["_load_error_"] = str(exc)
            params.pop("_t_data_", None)
            params.pop("_y_data_", None)
            return False

        # Sort by time so linear/searchsorted interpolation is well-defined even
        # if the file is unordered.
        order = np.argsort(t_data, kind="stable")
        params["_t_data_"] = t_data[order]
        params["_y_data_"] = y_data[order]
        logger.info("FromFile: loaded %d points from %s", t_data.size, data_file)
        return True

    @staticmethod
    def _sample(time, t_data, y_data, interpolation, end_behavior):
        """Sample the cached series at ``time`` given the interp/end policy."""
        t0, t1 = t_data[0], t_data[-1]

        if end_behavior == "loop" and t1 > t0 and (time < t0 or time > t1):
            # Wrap time into [t0, t1).
            span = t1 - t0
            time = t0 + ((time - t0) % span)

        if interpolation == "nearest":
            idx = int(np.argmin(np.abs(t_data - time)))
            return float(y_data[idx])

        if interpolation == "zoh":
            # Zero-order hold: value of the most recent sample at or before time.
            if time <= t0:
                return float(y_data[0])
            if time >= t1:
                return float(y_data[-1])
            idx = int(np.searchsorted(t_data, time, side="right")) - 1
            idx = max(0, min(idx, y_data.size - 1))
            return float(y_data[idx])

        # Linear (np.interp clamps to the edge values outside the range, which
        # is exactly the 'hold' policy for the in-range-after-wrap case).
        return float(np.interp(time, t_data, y_data))

    # ------------------------------------------------------------------ execute
    def execute(self, time, inputs, params, **kwargs):
        if not self._ensure_loaded(params):
            return {"E": True,
                    "error": params.get("_load_error_", "FromFile load failed")}

        t_data = params["_t_data_"]
        y_data = params["_y_data_"]
        value = self._sample(
            float(time), t_data, y_data,
            params.get("interpolation", "linear"),
            params.get("end_behavior", "hold"),
        )
        return {0: np.atleast_1d(np.array(value, dtype=float)), "E": False}
