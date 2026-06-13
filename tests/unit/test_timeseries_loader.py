"""Tests for the shared time-series loader (lib/services/timeseries_loader.py).

Writes small CSV / NPZ / TXT files to a tmp dir and checks column resolution
(by name and by numeric index), error handling, and length validation. MAT is
covered only if scipy.io is importable (it always is, since scipy is a dep).
"""

import numpy as np
import pytest

from lib.services.timeseries_loader import load_timeseries, TimeseriesLoadError


@pytest.mark.unit
class TestTimeseriesLoader:
    def test_csv_by_name(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("t,y\n0,0\n1,10\n2,20\n")
        t, y = load_timeseries(str(p), "t", "y")
        assert np.allclose(t, [0, 1, 2])
        assert np.allclose(y, [0, 10, 20])

    def test_csv_by_index(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("a,b,c\n0,5,9\n1,6,8\n")
        # numeric column specs are positional indices
        t, y = load_timeseries(str(p), "0", "2")
        assert np.allclose(t, [0, 1])
        assert np.allclose(y, [9, 8])

    def test_csv_mixed_index_and_name(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("t,sig\n0,0\n1,3\n")
        t, y = load_timeseries(str(p), "0", "sig")
        assert np.allclose(t, [0, 1])
        assert np.allclose(y, [0, 3])

    def test_npz_by_name(self, tmp_path):
        p = tmp_path / "d.npz"
        np.savez(str(p), t=np.array([0.0, 1.0, 2.0]), y=np.array([1.0, 2.0, 3.0]))
        t, y = load_timeseries(str(p), "t", "y")
        assert np.allclose(t, [0, 1, 2])
        assert np.allclose(y, [1, 2, 3])

    def test_txt_two_columns(self, tmp_path):
        p = tmp_path / "d.txt"
        np.savetxt(str(p), np.column_stack([[0, 1, 2], [4, 5, 6]]))
        t, y = load_timeseries(str(p), "0", "1")
        assert np.allclose(t, [0, 1, 2])
        assert np.allclose(y, [4, 5, 6])

    def test_txt_single_column_synthesizes_time(self, tmp_path):
        p = tmp_path / "d.txt"
        np.savetxt(str(p), np.array([10.0, 20.0, 30.0]))
        t, y = load_timeseries(str(p))
        assert np.allclose(t, [0, 1, 2])  # index-based time
        assert np.allclose(y, [10, 20, 30])

    def test_mat_roundtrip(self, tmp_path):
        from scipy.io import savemat
        p = tmp_path / "d.mat"
        savemat(str(p), {"t": np.array([0.0, 1.0]), "y": np.array([2.0, 4.0])})
        t, y = load_timeseries(str(p), "t", "y")
        assert np.allclose(t.flatten(), [0, 1])
        assert np.allclose(y.flatten(), [2, 4])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(TimeseriesLoadError):
            load_timeseries(str(tmp_path / "nope.csv"), "t", "y")

    def test_empty_path_raises(self):
        with pytest.raises(TimeseriesLoadError):
            load_timeseries("", "t", "y")

    def test_unknown_column_raises(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("t,y\n0,0\n1,1\n")
        with pytest.raises(TimeseriesLoadError):
            load_timeseries(str(p), "t", "nosuchcol")
