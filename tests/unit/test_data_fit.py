"""Spline-interpolation tests for the DataFit block (blocks/optimization/data_fit.py).

Regression focus: when SciPy's ``UnivariateSpline`` could not be built the block
fell back to linear interpolation inside a bare ``except Exception: pass``. The
run then silently used a different interpolation scheme than the user selected.
The fallback now logs a warning.

General DataFit coverage (ports, params, fit metrics, file loading) lives in
``test_data_fit_block.py``; this file only pins the spline path.
"""

import logging

import numpy as np
import pytest

from blocks.optimization.data_fit import DataFitBlock


def _params(**ov):
    p = {
        "data_file": "",
        "fit_type": "MSE",
        "weight": 1.0,
        "interpolation": "spline",
        "_init_start_": False,
        # Pre-loaded data, bypassing _load_data().
        "_time_data_": np.array([0.0, 1.0, 2.0, 3.0]),
        "_signal_data_": np.array([0.0, 1.0, 4.0, 9.0]),
        "_accumulated_error_": 0.0,
        "_n_points_": 0,
        "_ss_tot_": 0.0,
        "_ss_res_": 0.0,
    }
    p.update(ov)
    return p


@pytest.mark.unit
class TestDataFitSpline:
    def test_spline_interpolates_the_measured_signal(self):
        block = DataFitBlock()
        result = block.execute(1.0, {0: 0.0}, _params())

        assert result["E"] is False
        assert np.isclose(result[1], 1.0)

    def test_spline_is_built_once_and_cached(self):
        block = DataFitBlock()
        params = _params()

        block.execute(0.5, {0: 0.0}, params)
        first = params["_spline_"]
        block.execute(1.5, {0: 0.0}, params)

        assert params["_spline_"] is first


@pytest.mark.unit
class TestDataFitSplineFallback:
    def test_spline_failure_falls_back_to_linear_and_warns(self, monkeypatch, caplog):
        import scipy.interpolate

        def _boom(*args, **kwargs):
            raise RuntimeError("no spline for you")

        monkeypatch.setattr(scipy.interpolate, "UnivariateSpline", _boom)

        block = DataFitBlock()
        params = _params(_name_="DF1")

        with caplog.at_level(logging.WARNING, logger="blocks.optimization.data_fit"):
            result = block.execute(0.5, {0: 0.0}, params)

        # Fell back to linear interpolation ...
        assert np.isclose(result[1], 0.5)
        # ... and said so, instead of failing over in silence.
        messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
        assert any("spline" in m and "linear" in m for m in messages)
        assert any("DF1" in m for m in messages)

    def test_fallback_sentinel_stops_the_block_retrying_every_step(self, monkeypatch, caplog):
        import scipy.interpolate

        calls = []

        def _boom(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("no spline for you")

        monkeypatch.setattr(scipy.interpolate, "UnivariateSpline", _boom)

        block = DataFitBlock()
        params = _params()

        with caplog.at_level(logging.WARNING, logger="blocks.optimization.data_fit"):
            for k in range(5):
                block.execute(k * 0.1, {0: 0.0}, params)

        assert len(calls) == 1
        assert params["_spline_"] is np.interp
