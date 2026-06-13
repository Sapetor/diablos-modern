"""Shared time-series loader for data-driven blocks.

Loads a ``(time, signal)`` pair from a CSV / NPZ / MAT / whitespace-text file
and returns two 1-D float arrays. This is the single validated loader behind
both the :class:`FromFile` source block (signal replay) and the optimization
``DataFit`` block (reference signal), so the file-format handling lives in one
place instead of being duplicated per block.

Column resolution is uniform across formats: a *numeric* column spec is treated
as a positional index, anything else as a named lookup (CSV header / NPZ key /
MAT variable name). This supports mixed cases such as a numeric ``time_col``
with a named ``signal_col``.

Security: ``np.load`` is always called with ``allow_pickle=False`` so a data
file referenced from a (possibly untrusted) project file can never deserialize
arbitrary Python objects.
"""

import csv
import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["load_timeseries", "TimeseriesLoadError"]


class TimeseriesLoadError(ValueError):
    """Raised when a time-series file cannot be loaded or parsed."""


def _as_1d(arr):
    return np.atleast_1d(np.asarray(arr, dtype=float)).flatten()


def _column_from_rows(rows, col):
    """Resolve one column from a list of CSV ``DictReader`` rows.

    ``col`` is a positional index when it is all-digits, otherwise a header name.
    """
    if not rows:
        return np.array([], dtype=float)
    if str(col).isdigit():
        idx = int(col)
        return np.array([float(list(row.values())[idx]) for row in rows], dtype=float)
    if col not in rows[0]:
        raise TimeseriesLoadError(
            f"Column '{col}' not found (available: {sorted(rows[0].keys())})"
        )
    return np.array([float(row.get(col, 0.0)) for row in rows], dtype=float)


def _column_from_mapping(mapping, col, default_key):
    """Resolve one column from a dict-like {name: array} (NPZ / MAT)."""
    keys = [k for k in mapping.keys() if not str(k).startswith("__")]
    if str(col).isdigit():
        idx = int(col)
        if idx >= len(keys):
            raise TimeseriesLoadError(
                f"Column index {idx} out of range (file has {len(keys)} columns)"
            )
        return _as_1d(mapping[sorted(keys)[idx]])
    if col in mapping:
        return _as_1d(mapping[col])
    if default_key in mapping:
        return _as_1d(mapping[default_key])
    raise TimeseriesLoadError(
        f"Column '{col}' not found (available: {sorted(keys)})"
    )


def load_timeseries(data_file, time_col="t", signal_col="y"):
    """Load a ``(time, signal)`` series from ``data_file``.

    Args:
        data_file: path to a ``.csv`` / ``.npz`` / ``.mat`` / whitespace-text file.
        time_col: column index (numeric string/int) or name for the time axis.
        signal_col: column index or name for the signal.

    Returns:
        ``(t, y)`` — two 1-D float ``ndarray`` of equal length.

    Raises:
        TimeseriesLoadError: the file is missing, unreadable, or a requested
            column cannot be resolved.
    """
    if not data_file:
        raise TimeseriesLoadError("No data file specified")

    time_col = str(time_col)
    signal_col = str(signal_col)
    lower = str(data_file).lower()

    try:
        if lower.endswith(".npz"):
            # allow_pickle=False: never deserialize Python objects from a data
            # file path that may originate from an untrusted project file.
            with np.load(data_file, allow_pickle=False) as data:
                mapping = {k: data[k] for k in data.files}
            t_data = _column_from_mapping(mapping, time_col, "t")
            y_data = _column_from_mapping(mapping, signal_col, "y")

        elif lower.endswith(".mat"):
            from scipy.io import loadmat
            mapping = loadmat(data_file)
            t_data = _column_from_mapping(mapping, time_col, "t")
            y_data = _column_from_mapping(mapping, signal_col, "y")

        elif lower.endswith(".csv"):
            with open(data_file, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise TimeseriesLoadError("CSV file is empty")
            t_data = _column_from_rows(rows, time_col)
            y_data = _column_from_rows(rows, signal_col)

        else:
            # Whitespace-delimited text (.txt / .dat / unknown).
            data = np.loadtxt(data_file)
            if data.ndim == 1:
                # A single column is taken as the signal; synthesize an index time.
                y_data = _as_1d(data)
                t_data = np.arange(y_data.size, dtype=float)
            else:
                time_idx = int(time_col) if time_col.isdigit() else 0
                signal_idx = int(signal_col) if signal_col.isdigit() else 1
                t_data = _as_1d(data[:, time_idx])
                y_data = _as_1d(data[:, signal_idx])

    except TimeseriesLoadError:
        raise
    except FileNotFoundError as exc:
        raise TimeseriesLoadError(f"File not found: {data_file}") from exc
    except Exception as exc:  # malformed file, bad index, etc.
        raise TimeseriesLoadError(
            f"Failed to load '{data_file}': {exc}"
        ) from exc

    if t_data.size == 0 or y_data.size == 0:
        raise TimeseriesLoadError("Loaded data is empty")
    if t_data.size != y_data.size:
        raise TimeseriesLoadError(
            f"time column ({t_data.size}) and signal column ({y_data.size}) "
            "have different lengths"
        )
    return t_data, y_data
