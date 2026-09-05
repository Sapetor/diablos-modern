"""Unit tests for the DataFit calibration block (blocks/optimization/data_fit.py).

DataFit loads a measured (t, y) series and, at every step, reports how far the
simulation signal is from the measurement. The optimizer minimizes the value
``get_final_error()`` returns, so the pieces that matter are:

* the declared parameter/port contract (what the property editor offers);
* interpolation of the measured series onto the simulation clock, including the
  clamp outside the data range;
* the error metrics (MSE / MAE / RMSE) and the ``weight`` scaling;
* ``reset()`` re-arming the accumulators, without which optimizer iteration N+1
  inherits iteration N's error.

Kept to that documented surface on purpose: the block's internals are under
concurrent revision.
"""

import numpy as np
import pytest

from blocks.optimization.data_fit import DataFitBlock


def _defaults(**overrides):
    params = {name: spec["default"] for name, spec in DataFitBlock().params.items()}
    params.update(overrides)
    return params


def _write_csv(tmp_path, t_values, y_values, name="measured.csv"):
    path = tmp_path / name
    lines = ["t,y"] + [f"{t},{y}" for t, y in zip(t_values, y_values)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _drive(params, samples, dtime=0.1):
    """Feed ``[(time, signal), ...]``; return the (error, measured) pairs."""
    block = DataFitBlock()
    out = []
    for time, signal in samples:
        result = block.execute(
            time=time,
            inputs={0: np.array([float(signal)])},
            params=params,
            dtime=dtime,
        )
        assert result["E"] is False
        out.append((result[0], result[1]))
    return out


@pytest.mark.unit
class TestDataFitContract:
    def test_identity(self):
        block = DataFitBlock()
        assert block.block_name == "DataFit"
        assert block.category == "Optimization"

    def test_ports(self):
        block = DataFitBlock()
        assert [p["name"] for p in block.inputs] == ["signal"]
        assert [p["name"] for p in block.outputs] == ["error", "measured"]
        # Usually terminal: the error feeds the optimizer, not another block.
        assert block.requires_outputs is False

    def test_params(self):
        params = DataFitBlock().params
        assert set(params) == {
            "data_file",
            "time_col",
            "signal_col",
            "fit_type",
            "weight",
            "interpolation",
            "_init_start_",
        }
        assert params["fit_type"]["default"] == "MSE"
        assert params["time_col"]["default"] == "t"
        assert params["signal_col"]["default"] == "y"
        assert params["weight"]["default"] == 1.0
        assert params["_init_start_"]["default"] is True

    def test_draw_icon_returns_a_painter_path(self, qapp):
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QPainterPath

        path = DataFitBlock().draw_icon(QRect(0, 0, 100, 60))
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()


@pytest.mark.unit
class TestDataFitWithoutAFile:
    def test_no_data_file_is_not_an_error(self):
        """A freshly dropped block has ``data_file=''``; it must still run."""
        params = _defaults()
        results = _drive(params, [(0.0, 1.0), (0.1, 2.0)])

        assert params["_init_start_"] is False
        for error, measured in results:
            assert np.isfinite(error)
            assert measured == pytest.approx(0.0)

    def test_a_missing_file_falls_back_instead_of_raising(self, tmp_path):
        params = _defaults(data_file=str(tmp_path / "nope.csv"))
        results = _drive(params, [(0.0, 0.0)])

        assert results[0][1] == pytest.approx(0.0)


@pytest.mark.unit
class TestDataFitAgainstMeasuredData:
    def test_loads_the_series_and_interpolates_onto_the_sim_clock(self, tmp_path):
        # y = 2t sampled coarsely; the simulation asks for the midpoints.
        params = _defaults(data_file=_write_csv(tmp_path, [0.0, 1.0, 2.0], [0.0, 2.0, 4.0]))

        results = _drive(params, [(0.0, 0.0), (0.5, 0.0), (1.5, 0.0)])
        measured = [m for _, m in results]

        assert measured == pytest.approx([0.0, 1.0, 3.0])
        assert params["_time_data_"] == pytest.approx([0.0, 1.0, 2.0])
        assert params["_signal_data_"] == pytest.approx([0.0, 2.0, 4.0])

    def test_nearest_interpolation_snaps_to_a_sample(self, tmp_path):
        params = _defaults(
            data_file=_write_csv(tmp_path, [0.0, 1.0, 2.0], [10.0, 20.0, 30.0]),
            interpolation="nearest",
        )

        results = _drive(params, [(0.2, 0.0), (0.9, 0.0)])

        assert [m for _, m in results] == pytest.approx([10.0, 20.0])

    def test_time_outside_the_data_range_clamps_to_the_edge_values(self, tmp_path):
        params = _defaults(data_file=_write_csv(tmp_path, [1.0, 2.0], [5.0, 7.0]))

        results = _drive(params, [(0.0, 0.0), (99.0, 0.0)])

        assert [m for _, m in results] == pytest.approx([5.0, 7.0])

    def test_a_perfect_fit_gives_zero_error(self, tmp_path):
        params = _defaults(data_file=_write_csv(tmp_path, [0.0, 1.0, 2.0], [0.0, 2.0, 4.0]))

        results = _drive(params, [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)])

        for error, _ in results:
            assert error == pytest.approx(0.0)
        assert DataFitBlock().get_final_error(params) == pytest.approx(0.0)

    def test_mse_is_the_mean_squared_residual(self, tmp_path):
        params = _defaults(data_file=_write_csv(tmp_path, [0.0, 1.0], [0.0, 0.0]))

        # Residuals 1 and 3 -> MSE = (1 + 9) / 2 = 5
        _drive(params, [(0.0, 1.0), (1.0, 3.0)])

        assert DataFitBlock().get_final_error(params) == pytest.approx(5.0)

    def test_mae_is_the_mean_absolute_residual(self, tmp_path):
        params = _defaults(data_file=_write_csv(tmp_path, [0.0, 1.0], [0.0, 0.0]), fit_type="MAE")

        # Residuals -1 and 3 -> MAE = (1 + 3) / 2 = 2
        _drive(params, [(0.0, -1.0), (1.0, 3.0)])

        assert DataFitBlock().get_final_error(params) == pytest.approx(2.0)

    def test_rmse_is_the_square_root_of_mse(self, tmp_path):
        params = _defaults(data_file=_write_csv(tmp_path, [0.0, 1.0], [0.0, 0.0]), fit_type="RMSE")

        _drive(params, [(0.0, 1.0), (1.0, 3.0)])

        assert DataFitBlock().get_final_error(params) == pytest.approx(np.sqrt(5.0))

    def test_weight_scales_the_reported_error(self, tmp_path):
        csv_path = _write_csv(tmp_path, [0.0, 1.0], [0.0, 0.0])
        plain = _defaults(data_file=csv_path)
        weighted = _defaults(data_file=csv_path, weight=3.0)

        _drive(plain, [(0.0, 1.0), (1.0, 3.0)])
        _drive(weighted, [(0.0, 1.0), (1.0, 3.0)])

        block = DataFitBlock()
        assert block.get_final_error(weighted) == pytest.approx(3.0 * block.get_final_error(plain))


@pytest.mark.unit
class TestDataFitReset:
    def test_reset_clears_the_accumulators_for_the_next_iteration(self, tmp_path):
        """Without this, optimizer iteration N+1 starts with N's error."""
        block = DataFitBlock()
        params = _defaults(data_file=_write_csv(tmp_path, [0.0, 1.0], [0.0, 0.0]))
        _drive(params, [(0.0, 1.0), (1.0, 3.0)])
        assert block.get_final_error(params) > 0

        block.reset(params)

        assert params["_init_start_"] is True
        assert params["_accumulated_error_"] == 0.0
        assert params["_n_points_"] == 0

        # A second, perfect run must not inherit the first run's residuals.
        _drive(params, [(0.0, 0.0), (1.0, 0.0)])
        assert block.get_final_error(params) == pytest.approx(0.0)
