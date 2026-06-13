"""Tests for the FromFile source block (blocks/from_file.py).

Writes a small CSV and checks signal replay (linear / zoh / nearest), the
hold/loop end-behaviors, the load-once cache + reload-on-path-change, and the
error path for a missing file. State lives in params per the engine contract.
"""

import numpy as np
import pytest

from blocks.from_file import FromFileBlock


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "series.csv"
    p.write_text("t,y\n0,0\n1,10\n2,20\n")
    return str(p)


def _params(csv, **ov):
    p = {"data_file": csv, "time_col": "t", "signal_col": "y",
         "interpolation": "linear", "end_behavior": "hold"}
    p.update(ov)
    return p


@pytest.mark.unit
class TestFromFile:
    def setup_method(self):
        self.b = FromFileBlock()

    def test_linear_interpolation(self, csv_file):
        out = self.b.execute(0.5, {}, _params(csv_file))
        assert np.isclose(out[0][0], 5.0)

    def test_exact_sample(self, csv_file):
        out = self.b.execute(1.0, {}, _params(csv_file))
        assert np.isclose(out[0][0], 10.0)

    def test_hold_past_end(self, csv_file):
        out = self.b.execute(99.0, {}, _params(csv_file))
        assert np.isclose(out[0][0], 20.0)

    def test_hold_before_start(self, csv_file):
        out = self.b.execute(-5.0, {}, _params(csv_file))
        assert np.isclose(out[0][0], 0.0)

    def test_zoh(self, csv_file):
        # ZOH holds the most recent sample at-or-before the time.
        out = self.b.execute(1.9, {}, _params(csv_file, interpolation="zoh"))
        assert np.isclose(out[0][0], 10.0)

    def test_nearest(self, csv_file):
        out = self.b.execute(1.6, {}, _params(csv_file, interpolation="nearest"))
        assert np.isclose(out[0][0], 20.0)

    def test_loop(self, csv_file):
        # t=2.5 wraps into [0,2) -> 0.5 -> linear 5.0
        out = self.b.execute(2.5, {}, _params(csv_file, end_behavior="loop"))
        assert np.isclose(out[0][0], 5.0)

    def test_missing_file_errors(self, tmp_path):
        out = self.b.execute(0.0, {}, _params(str(tmp_path / "nope.csv")))
        assert out.get("E") is True

    def test_empty_path_errors(self):
        out = self.b.execute(0.0, {}, _params(""))
        assert out.get("E") is True

    def test_loads_once_then_caches(self, csv_file):
        params = _params(csv_file)
        self.b.execute(0.5, {}, params)
        # After the first call the data is cached and _init_start_ is cleared.
        assert "_t_data_" in params
        assert params.get("_init_start_") is False
        assert np.allclose(params["_t_data_"], [0, 1, 2])

    def test_reload_on_path_change(self, tmp_path, csv_file):
        params = _params(csv_file)
        self.b.execute(0.5, {}, params)
        # Point the same block at a different file mid-life; it must reload.
        other = tmp_path / "other.csv"
        other.write_text("t,y\n0,100\n1,200\n")
        params["data_file"] = str(other)
        out = self.b.execute(0.0, {}, params)
        assert np.isclose(out[0][0], 100.0)
        assert np.allclose(params["_y_data_"], [100, 200])

    def test_unsorted_file_is_sorted(self, tmp_path):
        p = tmp_path / "unsorted.csv"
        p.write_text("t,y\n2,20\n0,0\n1,10\n")
        out = self.b.execute(0.5, {}, _params(str(p)))
        assert np.isclose(out[0][0], 5.0)

    def test_is_source_no_inputs(self):
        assert self.b.requires_inputs is False
        assert self.b.inputs == []
        assert self.b.b_type == 0
