"""Headless tests for the Qt-free linearization exporter
(``lib/analysis/linearization_export.py``).

The formatter must round-trip A/B/C/D at FULL precision (repr-level, not the
display's %.4g), emit valid code in the empty-B/C/D and no-transfer-function
cases, and write .mat / .npz archives that load back with matching arrays.
"""

import sys
import types

import numpy as np
import pytest

from lib.analysis import linearization_export as lx


# --------------------------------------------------------------------- helpers
def _tricky_result():
    """A result whose entries would be mangled by %.4g formatting."""
    return {
        "ok": True,
        "n_states": 2,
        "state_names": ["x0", "x1"],
        "input_names": ["u0"],
        "output_names": ["y0"],
        "A": [[1.0 / 3.0, 1e-17], [-2.0, -3.0]],
        "B": [[0.0], [1.0]],
        "C": [[1.0, 0.0]],
        "D": [[0.0]],
        "poles": [[-1.0, 0.0], [-2.0, 0.0]],
        "zeros": [[-0.5, 0.0]],
        "tf_num": [1.0 / 7.0],
        "tf_den": [1.0, 3.0, 2.0],
    }


def _exec_python(code):
    """Exec generated Python with a stub ``control`` module; capture ss/tf args.

    Returns (namespace, calls) where calls is a dict with 'ss' and 'tf' entries
    holding the positional args passed (or None if never called).
    """
    calls = {"ss": None, "tf": None}

    stub = types.ModuleType("control")

    def _ss(*args):
        calls["ss"] = args
        return ("ss", args)

    def _tf(*args):
        calls["tf"] = args
        return ("tf", args)

    stub.ss = _ss
    stub.tf = _tf

    saved = sys.modules.get("control")
    sys.modules["control"] = stub
    try:
        ns = {}
        exec(compile(code, "<generated>", "exec"), ns)
    finally:
        if saved is None:
            sys.modules.pop("control", None)
        else:
            sys.modules["control"] = saved
    return ns, calls


# --------------------------------------------------------------- Python export
@pytest.mark.unit
class TestToPythonCode:
    def test_roundtrip_full_precision(self):
        result = _tricky_result()
        ns, calls = _exec_python(lx.to_python_code(result))

        assert np.array_equal(ns["A"], np.array(result["A"]))
        assert np.array_equal(ns["B"], np.array(result["B"]))
        assert np.array_equal(ns["C"], np.array(result["C"]))
        assert np.array_equal(ns["D"], np.array(result["D"]))

        # ss called with exactly the reconstructed A/B/C/D.
        assert calls["ss"] is not None
        a, b, c, d = calls["ss"]
        assert np.array_equal(a, np.array(result["A"]))
        assert np.array_equal(d, np.array(result["D"]))

        # tf present -> tf called; num/den at full precision.
        assert calls["tf"] is not None
        num, den = calls["tf"]
        assert np.array_equal(np.asarray(num), np.array(result["tf_num"]))
        assert np.array_equal(np.asarray(den), np.array(result["tf_den"]))

    def test_tiny_value_not_mangled(self):
        # 1e-17 would render as 0 under %.4g; ensure it survives exactly.
        ns, _ = _exec_python(lx.to_python_code(_tricky_result()))
        assert ns["A"][0, 1] == 1e-17
        assert ns["A"][0, 0] == 1.0 / 3.0

    def test_header_and_names_comment(self):
        code = lx.to_python_code(_tricky_result())
        assert "source: DiaBloS linearization" in code
        assert "import numpy as np" in code
        assert "import control" in code
        assert "x0, x1" in code  # state names as a comment
        assert "u0" in code

    def test_empty_bcd_no_ss(self):
        result = {
            "ok": True,
            "n_states": 1,
            "state_names": ["x0"],
            "A": [[-3.0]],
            "B": [],
            "C": [],
            "D": [],
            "poles": [[-3.0, 0.0]],
            "tf_num": None,
            "tf_den": None,
        }
        code = lx.to_python_code(result)
        ns, calls = _exec_python(code)
        assert np.array_equal(ns["A"], np.array([[-3.0]]))
        assert calls["ss"] is None  # no B/C -> no ss call
        assert calls["tf"] is None  # tf absent -> no tf call
        assert "control.ss(" not in code

    def test_synthesized_d_when_empty(self):
        result = _tricky_result()
        result["D"] = []
        ns, calls = _exec_python(lx.to_python_code(result))
        assert calls["ss"] is not None
        _, b, c, d = calls["ss"]
        # zero D of shape (n_outputs, n_inputs) = (C.rows, B.cols) = (1, 1).
        assert np.array_equal(np.asarray(d), np.zeros((1, 1)))

    def test_empty_everything_valid(self):
        result = {"ok": True, "n_states": 0, "A": [], "B": [], "C": [], "D": []}
        code = lx.to_python_code(result)
        ns, calls = _exec_python(code)
        assert calls["ss"] is None
        assert calls["tf"] is None
        assert "A = np.array([])" in code

    def test_empty_tf_lists_no_tf(self):
        # tf_num == [] (not None) must not emit an invalid control.tf([], den).
        result = _tricky_result()
        result["tf_num"] = []
        code = lx.to_python_code(result)
        ns, calls = _exec_python(code)
        assert calls["tf"] is None
        assert "control.tf(" not in code

    def test_no_control_import_when_unused(self):
        # A degenerate snippet with neither ss() nor tf() must run without
        # python-control installed, so the import is omitted entirely.
        result = {"ok": True, "n_states": 0, "A": [], "B": [], "C": [], "D": []}
        code = lx.to_python_code(result)
        assert "import control" not in code
        exec(compile(code, "<snippet>", "exec"), {})

    def test_nan_inf_do_not_crash(self):
        result = {
            "ok": True,
            "A": [[float("nan"), float("inf")], [float("-inf"), 1.0]],
            "B": [[1.0], [0.0]],
            "C": [[1.0, 0.0]],
            "D": [],
        }
        ns, calls = _exec_python(lx.to_python_code(result))
        assert np.isnan(ns["A"][0, 0])
        assert np.isposinf(ns["A"][0, 1])
        assert np.isneginf(ns["A"][1, 0])


# --------------------------------------------------------------- MATLAB export
@pytest.mark.unit
class TestToMatlabCode:
    def test_row_separators_and_precision(self):
        code = lx.to_matlab_code(_tricky_result())
        # A has two rows -> exactly one ';' inside the A literal.
        a_line = next(ln for ln in code.splitlines() if ln.startswith("A = "))
        assert a_line.count(";") == 2  # one row separator + trailing statement ';'
        # Full precision value present (1/3), not the %.4g truncation.
        assert repr(1.0 / 3.0) in code
        assert "1e-17" in code
        assert a_line.endswith(";")

    def test_ss_and_tf_lines(self):
        code = lx.to_matlab_code(_tricky_result())
        assert "sys = ss(A, B, C, D);" in code
        assert "G = tf(num, den);" in code
        assert code.lstrip().startswith("%")  # comment header

    def test_empty_bcd_no_ss_no_tf(self):
        result = {
            "ok": True,
            "A": [[-3.0]],
            "B": [],
            "C": [],
            "D": [],
            "tf_num": None,
            "tf_den": None,
        }
        code = lx.to_matlab_code(result)
        assert "ss(A, B, C, D)" not in code
        assert "tf(num, den)" not in code
        assert "A = [-3.0];" in code

    def test_nan_inf_tokens(self):
        result = {"ok": True, "A": [[float("nan"), float("inf")], [float("-inf"), 1.0]]}
        code = lx.to_matlab_code(result)
        assert "NaN" in code and "Inf" in code and "-Inf" in code

    def test_empty_tf_lists_no_tf(self):
        # tf_den == [] (not None) must not emit an invalid tf(num, []).
        result = _tricky_result()
        result["tf_den"] = []
        code = lx.to_matlab_code(result)
        assert "tf(num, den)" not in code


# ----------------------------------------------------------------- data export
@pytest.mark.unit
class TestSaveData:
    def test_save_mat(self, tmp_path):
        from scipy.io import loadmat

        result = _tricky_result()
        out = tmp_path / "model.mat"
        assert lx.save_data(result, str(out)) is True
        loaded = loadmat(str(out))
        assert np.array_equal(loaded["A"], np.array(result["A"]))
        assert np.array_equal(loaded["B"], np.array(result["B"]))
        assert "tf_num" in loaded and "poles" in loaded

    def test_save_npz(self, tmp_path):
        result = _tricky_result()
        out = tmp_path / "model.npz"
        assert lx.save_data(result, str(out)) is True
        with np.load(str(out)) as data:
            assert np.array_equal(data["A"], np.array(result["A"]))
            assert np.array_equal(data["C"], np.array(result["C"]))
            assert np.array_equal(data["tf_den"], np.array(result["tf_den"]))

    def test_save_skips_none_and_empties(self, tmp_path):
        result = {
            "ok": True,
            "A": [[-3.0]],
            "B": [],  # empty -> empty array, still written
            "C": None,  # None -> skipped
            "tf_num": None,  # skipped
        }
        out = tmp_path / "min.npz"
        lx.save_data(result, str(out))
        with np.load(str(out)) as data:
            assert "A" in data
            assert "B" in data and data["B"].size == 0
            assert "C" not in data
            assert "tf_num" not in data
