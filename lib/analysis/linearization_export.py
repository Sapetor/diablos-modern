"""Serialize a linearization result dict to portable code / data files.

Qt-free, pure numpy/scipy so it stays headless-testable and reusable by a CLI.
It CONSUMES the shared linearization result-dict contract (see
``modern_ui/widgets/linearization_result_window.py`` for the full contract) and
only reads the arrays already present in it -- it performs no linearization.

Three entry points:

  * :func:`to_python_code`  -- a self-contained ``numpy``/``python-control``
    snippet reconstructing the state-space model (and transfer function).
  * :func:`to_matlab_code`  -- the MATLAB equivalent.
  * :func:`save_data`       -- write A/B/C/D, tf_num/tf_den, poles, zeros to a
    ``.mat`` (``scipy.io.savemat``) or ``.npz`` (``np.savez``) archive.

All numeric output is at full round-trip precision (``repr`` of floats), NOT the
display's ``%.4g`` -- so exported values equal the originals exactly.
"""

import math
import os

import numpy as np


# --------------------------------------------------------------- float helpers
def _py_float(x):
    """Full-precision, exec-safe Python literal for a single float."""
    f = float(x)
    if math.isnan(f):
        return "np.nan"
    if math.isinf(f):
        return "np.inf" if f > 0 else "-np.inf"
    return repr(f)


def _matlab_float(x):
    """Full-precision MATLAB literal for a single float."""
    f = float(x)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "Inf" if f > 0 else "-Inf"
    return repr(f)


def _to_2d(mat):
    """Coerce a possibly-empty/None matrix into a 2-D float ndarray."""
    return np.atleast_2d(np.asarray([] if mat is None else mat, dtype=float))


def _to_1d(vec):
    """Coerce a coefficient list into a 1-D float ndarray (empty if None/[])."""
    return np.asarray([] if vec is None else vec, dtype=float).ravel()


# ---------------------------------------------------------------- code emitters
def _np_matrix(arr):
    """Render a 2-D ndarray as a full-precision ``np.array([...])`` literal."""
    if arr.size == 0:
        return "np.array([])"
    rows = ["[" + ", ".join(_py_float(v) for v in row) + "]" for row in arr]
    return "np.array([" + ", ".join(rows) + "])"


def _np_vector(arr):
    """Render a 1-D ndarray as a full-precision Python list literal."""
    return "[" + ", ".join(_py_float(v) for v in arr) + "]"


def _matlab_matrix(arr):
    """Render a 2-D ndarray as a MATLAB ``[a b; c d]`` literal."""
    if arr.size == 0:
        return "[]"
    rows = [" ".join(_matlab_float(v) for v in row) for row in arr]
    return "[" + "; ".join(rows) + "]"


def _matlab_vector(arr):
    """Render a 1-D ndarray as a MATLAB row-vector literal."""
    return "[" + " ".join(_matlab_float(v) for v in arr) + "]"


def _names_comment(result, prefix):
    """Emit ``<prefix> states/inputs/outputs`` comment lines (skip empties)."""
    lines = []
    for label, key in (
        ("states", "state_names"),
        ("inputs", "input_names"),
        ("outputs", "output_names"),
    ):
        names = result.get(key) or []
        if names:
            lines.append("{0} {1}:  {2}".format(prefix, label, ", ".join(str(n) for n in names)))
    return lines


def _pole_summary(result):
    """A compact ``re+imj`` list of the poles (for a comment)."""
    poles = result.get("poles") or []
    parts = []
    for p in poles:
        try:
            re = float(p[0])
            im = float(p[1]) if len(p) > 1 else 0.0
        except (TypeError, ValueError, IndexError):
            continue
        if im == 0.0:
            parts.append(_py_float(re))
        else:
            sign = "+" if im >= 0 else "-"
            parts.append("{0}{1}{2}j".format(_py_float(re), sign, _py_float(abs(im))))
    return ", ".join(parts)


def _resolve_ss(result):
    """Return ``(A, B, C, D, ss_ok)`` with D synthesized when omittable.

    ``ss_ok`` is True only when A, B and C are all non-empty. When it is True but
    D is empty, D is replaced by a zero matrix of the right (n_outputs, n_inputs)
    shape so the emitted ``ss(...)`` call is well-formed.
    """
    a = _to_2d(result.get("A"))
    b = _to_2d(result.get("B"))
    c = _to_2d(result.get("C"))
    d = _to_2d(result.get("D"))
    ss_ok = a.size > 0 and b.size > 0 and c.size > 0
    if ss_ok and d.size == 0:
        d = np.zeros((c.shape[0], b.shape[1]), dtype=float)
    return a, b, c, d, ss_ok


# ------------------------------------------------------------- code emission
def _emit_code(result, py):
    """Shared emitter skeleton; ``py`` selects Python vs MATLAB dialect."""
    result = result or {}
    a, b, c, d, ss_ok = _resolve_ss(result)
    num = _to_1d(result.get("tf_num"))
    den = _to_1d(result.get("tf_den"))
    tf_ok = num.size > 0 and den.size > 0
    cm = "#" if py else "%"
    fmt_matrix = _np_matrix if py else _matlab_matrix
    fmt_vector = _np_vector if py else _matlab_vector
    sfx = "" if py else ";"

    lines = [cm + " Linearized state-space model", cm + " source: DiaBloS linearization"]
    lines.extend(_names_comment(result, cm))
    poles = _pole_summary(result)
    if poles:
        lines.append(cm + " poles: " + poles)
    lines.append("")
    if py:
        lines.append("import numpy as np")
        if ss_ok or tf_ok:
            lines.append("import control")
        lines.append("")
    for name, mat in (("A", a), ("B", b), ("C", c), ("D", d)):
        lines.append(name + " = " + fmt_matrix(mat) + sfx)
    if ss_ok:
        lines.append("sys = control.ss(A, B, C, D)" if py else "sys = ss(A, B, C, D);")

    if tf_ok:
        lines.append("")
        lines.append(cm + " Transfer function")
        lines.append("num = " + fmt_vector(num) + sfx)
        lines.append("den = " + fmt_vector(den) + sfx)
        lines.append("G = control.tf(num, den)" if py else "G = tf(num, den);")

    return "\n".join(lines) + "\n"


def to_python_code(result):
    """Return a self-contained numpy / python-control snippet as a string."""
    return _emit_code(result, py=True)


def to_matlab_code(result):
    """Return a self-contained MATLAB snippet as a string."""
    return _emit_code(result, py=False)


# ----------------------------------------------------------------- data export
def _collect_arrays(result):
    """Build the ``{name: ndarray}`` map written to a .mat/.npz archive.

    Absent / ``None`` entries are skipped; empty lists become empty arrays.
    """
    result = result or {}
    data = {}
    for key, coerce in (
        ("A", _to_2d),
        ("B", _to_2d),
        ("C", _to_2d),
        ("D", _to_2d),
        ("tf_num", _to_1d),
        ("tf_den", _to_1d),
        ("poles", lambda v: np.asarray(v, dtype=float)),
        ("zeros", lambda v: np.asarray(v, dtype=float)),
    ):
        value = result.get(key)
        if value is not None:
            data[key] = coerce(value)
    return data


def save_data(result, path):
    """Write the model arrays to ``path`` as a ``.mat`` or ``.npz`` archive.

    Format is chosen from the extension (``.npz`` -> numpy archive, anything else
    -> MAT-file). Returns True on success.
    """
    data = _collect_arrays(result)
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".npz":
        np.savez(path, **data)
    else:
        from scipy.io import savemat

        savemat(path, data)
    return True
