"""1-D and 2-D Lookup Table blocks (interpolation).

Maps an input through a user-supplied breakpoint/value table using linear or
nearest interpolation, with selectable extrapolation (clip vs. linear). Lookup
tables are the standard way to model nonlinear actuator/sensor curves,
gain-scheduled controllers, valve characteristics, and empirically measured
plant maps.

Table parameters are entered as text and parsed with
:func:`lib.safe_eval.safe_literal` (literal lists only — no code execution),
matching the ``MatrixGain`` block's matrix-parameter convention.

Both blocks run on the interpreter path (no compiled executor yet).
"""

import logging

import numpy as np

from blocks.base_block import BaseBlock
from lib.safe_eval import safe_literal, SafeEvalError

logger = logging.getLogger(__name__)


def _parse_array(raw, name):
    """Parse a table parameter (string literal or array) into a float ndarray.

    Raises:
        ValueError: if the parameter cannot be parsed as a numeric list. It used
            to fall back to a hard-coded default table, which silently replaced
            the user's curve with a straight line and falsified the run; the
            caller turns this into an ``{'E': True, 'error': ...}`` result.
    """
    if raw is None:
        raise ValueError(f"{name} is not set")
    try:
        value = safe_literal(raw) if isinstance(raw, str) else raw
        return np.asarray(value, dtype=float)
    except (SafeEvalError, ValueError, TypeError) as exc:
        logger.warning("Lookup table parameter %s could not be parsed from %r: %s", name, raw, exc)
        raise ValueError(f"{name}: cannot parse {raw!r} as a numeric list ({exc})") from exc


def _table_key(*arrays):
    """Hashable fingerprint of the parsed table arrays, for the interpolator cache."""
    return tuple((a.shape, a.tobytes()) for a in arrays)


class LookupTable1DBlock(BaseBlock):
    """One-dimensional lookup table: y = f(x) by interpolating a breakpoint table."""

    @property
    def block_name(self):
        return "LookupTable1D"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "darkcyan"

    @property
    def doc(self):
        return (
            "1-D lookup table: maps the input x through a breakpoint/value table "
            "by interpolation.\n\n"
            "Parameters:\n"
            "- x_values: breakpoints, e.g. [0, 1, 2, 3] (must be distinct).\n"
            "- y_values: table values, same length as x_values.\n"
            "- interpolation: 'linear' or 'nearest'.\n"
            "- extrapolation: 'clip' (hold the edge value) or 'linear' "
            "(extend the end slope) outside the breakpoint range.\n\n"
            "Usage:\nModel nonlinear sensor/actuator curves or measured maps."
        )

    @property
    def params(self):
        return {
            "x_values": {
                "type": "string",
                "default": "[0, 1, 2, 3]",
                "doc": "Breakpoints (distinct), e.g. [0, 1, 2, 3].",
            },
            "y_values": {
                "type": "string",
                "default": "[0, 1, 4, 9]",
                "doc": "Table values, same length as x_values.",
            },
            "interpolation": {
                "type": "choice",
                "default": "linear",
                "options": ["linear", "nearest"],
                "doc": "Interpolation method.",
            },
            "extrapolation": {
                "type": "choice",
                "default": "clip",
                "options": ["clip", "linear"],
                "doc": "Outside the table: clip (hold edge) or linear (extend).",
            },
        }

    @property
    def inputs(self):
        return [{"name": "x", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw a monotone staircase/curve glyph in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        path.moveTo(0.15, 0.80)
        path.lineTo(0.35, 0.80)
        path.lineTo(0.55, 0.45)
        path.lineTo(0.75, 0.25)
        path.lineTo(0.88, 0.20)
        return path

    def execute(self, time, inputs, params, **kwargs):
        try:
            from scipy.interpolate import interp1d

            x_in = np.atleast_1d(inputs.get(0, 0.0)).astype(float)
            xv = _parse_array(params.get("x_values"), "x_values").flatten()
            yv = _parse_array(params.get("y_values"), "y_values").flatten()

            if xv.size != yv.size:
                return {
                    "E": True,
                    "error": (
                        f"x_values ({xv.size}) and y_values ({yv.size}) must have the same length"
                    ),
                }
            if xv.size < 2:
                return {"E": True, "error": "Lookup table needs at least 2 breakpoints"}

            # Sort by x so edge fill-values map to the true min/max breakpoints.
            order = np.argsort(xv, kind="stable")
            xv, yv = xv[order], yv[order]
            if np.any(np.diff(xv) == 0):
                return {"E": True, "error": "x_values must be distinct (no repeats)"}

            kind = params.get("interpolation", "linear")
            extrap = params.get("extrapolation", "clip")

            # Cache the interpolator: building it is far more expensive than
            # evaluating it, and the table is constant for the whole run in all
            # but the (rare) workspace-variable case. The key covers the table
            # contents and both mode strings, so an edited table rebuilds it.
            cache_key = (_table_key(xv, yv), kind, extrap)
            f = params.get("_interp1d_") if params.get("_interp1d_key_") == cache_key else None
            if f is None:
                fill = "extrapolate" if extrap == "linear" else (yv[0], yv[-1])
                f = interp1d(
                    xv, yv, kind=kind, bounds_error=False, fill_value=fill, assume_sorted=True
                )
                params["_interp1d_"] = f
                params["_interp1d_key_"] = cache_key
            # interp1d on a 1-D input returns a 1-D array (scalar in -> size-1).
            return {0: np.asarray(f(x_in), dtype=float)}
        except (ValueError, TypeError) as exc:
            logger.error("LookupTable1D error: %s", exc)
            return {"E": True, "error": str(exc)}


class LookupTable2DBlock(BaseBlock):
    """Two-dimensional lookup table: z = f(x, y) over a regular (x, y) grid."""

    @property
    def block_name(self):
        return "LookupTable2D"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "darkcyan"

    @property
    def doc(self):
        return (
            "2-D lookup table: maps inputs (x, y) through a regular grid of "
            "values by interpolation.\n\n"
            "Parameters:\n"
            "- x_values: row breakpoints, e.g. [0, 1, 2].\n"
            "- y_values: column breakpoints, e.g. [0, 1].\n"
            "- z_table: values as a list of rows with shape "
            "[len(x_values)][len(y_values)], e.g. [[0,1],[2,3],[4,5]].\n"
            "- interpolation: 'linear' or 'nearest'.\n"
            "- extrapolation: 'clip' (hold edge) or 'linear' (extend).\n\n"
            "Usage:\nGain scheduling and 2-D actuator/plant maps."
        )

    @property
    def params(self):
        return {
            "x_values": {
                "type": "string",
                "default": "[0, 1, 2]",
                "doc": "Row breakpoints (distinct).",
            },
            "y_values": {
                "type": "string",
                "default": "[0, 1]",
                "doc": "Column breakpoints (distinct).",
            },
            "z_table": {
                "type": "string",
                "default": "[[0, 1], [2, 3], [4, 5]]",
                "doc": "Values, shape [len(x_values)][len(y_values)].",
            },
            "interpolation": {
                "type": "choice",
                "default": "linear",
                "options": ["linear", "nearest"],
                "doc": "Interpolation method.",
            },
            "extrapolation": {
                "type": "choice",
                "default": "clip",
                "options": ["clip", "linear"],
                "doc": "Outside the grid: clip (hold edge) or linear (extend).",
            },
        }

    @property
    def inputs(self):
        return [{"name": "x", "type": "any"}, {"name": "y", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "z", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw a small grid glyph in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        for gx in (0.25, 0.45, 0.65, 0.85):
            path.moveTo(gx, 0.18)
            path.lineTo(gx, 0.82)
        for gy in (0.25, 0.45, 0.65, 0.82):
            path.moveTo(0.25, gy)
            path.lineTo(0.85, gy)
        return path

    def execute(self, time, inputs, params, **kwargs):
        try:
            from scipy.interpolate import RegularGridInterpolator

            xv = _parse_array(params.get("x_values"), "x_values").flatten()
            yv = _parse_array(params.get("y_values"), "y_values").flatten()
            Z = _parse_array(params.get("z_table"), "z_table")

            if xv.size < 2 or yv.size < 2:
                return {"E": True, "error": "Each axis needs at least 2 breakpoints"}
            Z = np.atleast_2d(Z)
            if Z.shape != (xv.size, yv.size):
                return {
                    "E": True,
                    "error": (
                        f"z_table shape {Z.shape} must be "
                        f"({xv.size}, {yv.size}) = (len(x_values), len(y_values))"
                    ),
                }

            # Sort each axis ascending and reorder Z to match.
            xo = np.argsort(xv, kind="stable")
            yo = np.argsort(yv, kind="stable")
            xv, yv, Z = xv[xo], yv[yo], Z[np.ix_(xo, yo)]
            if np.any(np.diff(xv) == 0) or np.any(np.diff(yv) == 0):
                return {"E": True, "error": "x_values and y_values must each be distinct"}

            method = params.get("interpolation", "linear")
            extrap = params.get("extrapolation", "clip")

            # Cache the interpolator (see LookupTable1D): rebuilding it on every
            # time step dominates the cost of a table lookup.
            cache_key = (_table_key(xv, yv, Z), method, extrap)
            rgi = params.get("_rgi_") if params.get("_rgi_key_") == cache_key else None
            if rgi is None:
                fill_value = None if extrap == "linear" else np.nan
                rgi = RegularGridInterpolator(
                    (xv, yv), Z, method=method, bounds_error=False, fill_value=fill_value
                )
                params["_rgi_"] = rgi
                params["_rgi_key_"] = cache_key

            x_in = np.atleast_1d(inputs.get(0, 0.0)).astype(float)
            y_in = np.atleast_1d(inputs.get(1, 0.0)).astype(float)
            x_in, y_in = np.broadcast_arrays(x_in, y_in)

            if extrap == "clip":
                x_in = np.clip(x_in, xv[0], xv[-1])
                y_in = np.clip(y_in, yv[0], yv[-1])

            pts = np.column_stack([x_in.ravel(), y_in.ravel()])
            z_out = np.asarray(rgi(pts), dtype=float)
            return {0: float(z_out[0]) if z_out.size == 1 else z_out}
        except (ValueError, TypeError) as exc:
            logger.error("LookupTable2D error: %s", exc)
            return {"E": True, "error": str(exc)}
