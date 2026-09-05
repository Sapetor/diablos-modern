"""Tests for the 1-D / 2-D Lookup Table blocks (blocks/lookup_table.py).

Instantiates each block and calls execute() with hand-chosen tables so the
interpolated/extrapolated outputs are exactly predictable.
"""

import logging

import numpy as np
import pytest

from blocks.lookup_table import LookupTable1DBlock, LookupTable2DBlock


def _p1(**ov):
    p = {
        "x_values": "[0, 1, 2, 3]",
        "y_values": "[0, 1, 4, 9]",
        "interpolation": "linear",
        "extrapolation": "clip",
    }
    p.update(ov)
    return p


def _p2(**ov):
    p = {
        "x_values": "[0, 1, 2]",
        "y_values": "[0, 1]",
        "z_table": "[[0, 1], [2, 3], [4, 5]]",
        "interpolation": "linear",
        "extrapolation": "clip",
    }
    p.update(ov)
    return p


@pytest.mark.unit
class TestLookupTable1D:
    def setup_method(self):
        self.b = LookupTable1DBlock()

    def test_linear_interpolation(self):
        out = self.b.execute(0.0, {0: np.array([1.5])}, _p1())
        assert np.isclose(out[0][0], 2.5)  # between y=1 and y=4

    def test_exact_breakpoint(self):
        out = self.b.execute(0.0, {0: np.array([2.0])}, _p1())
        assert np.isclose(out[0][0], 4.0)

    def test_clip_extrapolation(self):
        below = self.b.execute(0.0, {0: np.array([-5.0])}, _p1())
        above = self.b.execute(0.0, {0: np.array([10.0])}, _p1())
        assert np.isclose(below[0][0], 0.0)  # held at y[0]
        assert np.isclose(above[0][0], 9.0)  # held at y[-1]

    def test_linear_extrapolation(self):
        out = self.b.execute(0.0, {0: np.array([4.0])}, _p1(extrapolation="linear"))
        assert np.isclose(out[0][0], 14.0)  # last slope 5 from (2,4)->(3,9)

    def test_nearest(self):
        lo = self.b.execute(0.0, {0: np.array([1.4])}, _p1(interpolation="nearest"))
        hi = self.b.execute(0.0, {0: np.array([1.6])}, _p1(interpolation="nearest"))
        assert np.isclose(lo[0][0], 1.0)
        assert np.isclose(hi[0][0], 4.0)

    def test_unsorted_breakpoints_are_sorted(self):
        out = self.b.execute(
            0.0, {0: np.array([1.5])}, _p1(x_values="[3, 0, 2, 1]", y_values="[9, 0, 4, 1]")
        )
        assert np.isclose(out[0][0], 2.5)

    def test_length_mismatch_errors(self):
        out = self.b.execute(0.0, {0: 1.0}, _p1(x_values="[0, 1]", y_values="[0, 1, 2]"))
        assert out.get("E") is True

    def test_too_few_points_errors(self):
        out = self.b.execute(0.0, {0: 1.0}, _p1(x_values="[0]", y_values="[5]"))
        assert out.get("E") is True

    def test_duplicate_breakpoints_error(self):
        out = self.b.execute(0.0, {0: 1.0}, _p1(x_values="[0, 1, 1, 2]", y_values="[0, 1, 2, 3]"))
        assert out.get("E") is True

    def test_vector_input(self):
        out = self.b.execute(0.0, {0: np.array([0.5, 2.5])}, _p1())
        assert np.allclose(out[0], [0.5, 6.5])


@pytest.mark.unit
class TestLookupTable2D:
    def setup_method(self):
        self.b = LookupTable2DBlock()

    def test_grid_corner(self):
        # Z[i, j] over x in {0,1,2}, y in {0,1}; Z[1,0]=2
        out = self.b.execute(0.0, {0: 1.0, 1: 0.0}, _p2())
        assert np.isclose(out[0], 2.0)

    def test_bilinear_center(self):
        # corners 0,1,2,3 around (x in [0,1], y in [0,1]); center -> mean=1.5
        out = self.b.execute(0.0, {0: 0.5, 1: 0.5}, _p2())
        assert np.isclose(out[0], 1.5)

    def test_clip(self):
        out = self.b.execute(0.0, {0: 9.0, 1: 9.0}, _p2())
        assert np.isclose(out[0], 5.0)  # Z[2,1]

    def test_nearest(self):
        out = self.b.execute(0.0, {0: 1.9, 1: 0.1}, _p2(interpolation="nearest"))
        assert np.isclose(out[0], 4.0)  # nearest grid point Z[2,0]

    def test_shape_mismatch_errors(self):
        out = self.b.execute(0.0, {0: 0.0, 1: 0.0}, _p2(z_table="[[0, 1]]"))
        assert out.get("E") is True

    def test_too_few_breakpoints_error(self):
        out = self.b.execute(0.0, {0: 0.0, 1: 0.0}, _p2(x_values="[0]", z_table="[[0, 1]]"))
        assert out.get("E") is True


@pytest.mark.unit
class TestLookupTableParameterErrors:
    """Regression: unparsable table parameters were silently replaced.

    ``_parse_array`` used to swallow the parse error and substitute a hard-coded
    default table ([0, 1] / [[0,0],[0,0]]), so a typo in the breakpoints turned
    the user's nonlinear curve into a straight line with nothing in the log.
    """

    @pytest.mark.parametrize("bad", ["[0, 1, ", "not a list", "[0, 'a', 2]", ""])
    def test_1d_bad_x_values_returns_an_error_dict(self, bad):
        from blocks.lookup_table import LookupTable1DBlock

        block = LookupTable1DBlock()
        params = {"x_values": bad, "y_values": "[0, 1, 4, 9]", "interpolation": "linear"}
        result = block.execute(0.0, {0: 1.0}, params)

        assert result.get("E") is True
        assert "x_values" in result["error"]
        assert 0 not in result

    def test_1d_bad_y_values_returns_an_error_dict(self):
        from blocks.lookup_table import LookupTable1DBlock

        block = LookupTable1DBlock()
        params = {"x_values": "[0, 1, 2, 3]", "y_values": "[0, 1,", "interpolation": "linear"}
        result = block.execute(0.0, {0: 1.0}, params)

        assert result.get("E") is True
        assert "y_values" in result["error"]

    def test_1d_missing_parameter_returns_an_error_dict(self):
        from blocks.lookup_table import LookupTable1DBlock

        block = LookupTable1DBlock()
        result = block.execute(0.0, {0: 1.0}, {"y_values": "[0, 1]"})

        assert result.get("E") is True
        assert "x_values" in result["error"]

    def test_1d_parse_failure_is_logged(self, caplog):
        from blocks.lookup_table import LookupTable1DBlock

        block = LookupTable1DBlock()
        with caplog.at_level(logging.WARNING, logger="blocks.lookup_table"):
            block.execute(0.0, {0: 1.0}, {"x_values": "oops", "y_values": "[0, 1]"})

        assert any("x_values" in rec.getMessage() for rec in caplog.records)
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_2d_bad_z_table_returns_an_error_dict(self):
        from blocks.lookup_table import LookupTable2DBlock

        block = LookupTable2DBlock()
        params = {"x_values": "[0, 1, 2]", "y_values": "[0, 1]", "z_table": "[[0, 1], [2,"}
        result = block.execute(0.0, {0: 0.5, 1: 0.5}, params)

        assert result.get("E") is True
        assert "z_table" in result["error"]


@pytest.mark.unit
class TestLookupTableInterpolatorCache:
    """The interpolator is rebuilt only when the table (or a mode) changes."""

    def test_1d_interpolator_is_reused_across_steps(self):
        from blocks.lookup_table import LookupTable1DBlock

        block = LookupTable1DBlock()
        params = {"x_values": "[0, 1, 2, 3]", "y_values": "[0, 1, 4, 9]"}

        block.execute(0.0, {0: 1.5}, params)
        first = params["_interp1d_"]
        block.execute(0.1, {0: 2.5}, params)

        assert params["_interp1d_"] is first

    def test_1d_interpolator_is_rebuilt_when_the_table_changes(self):
        from blocks.lookup_table import LookupTable1DBlock

        block = LookupTable1DBlock()
        params = {"x_values": "[0, 1, 2, 3]", "y_values": "[0, 1, 4, 9]"}

        first_out = block.execute(0.0, {0: 1.5}, params)[0]
        first = params["_interp1d_"]

        params["y_values"] = "[0, 10, 40, 90]"
        second_out = block.execute(0.1, {0: 1.5}, params)[0]

        assert params["_interp1d_"] is not first
        np.testing.assert_allclose(second_out, np.asarray(first_out) * 10.0)

    def test_1d_interpolator_is_rebuilt_when_the_mode_changes(self):
        from blocks.lookup_table import LookupTable1DBlock

        block = LookupTable1DBlock()
        params = {"x_values": "[0, 1, 2, 3]", "y_values": "[0, 1, 4, 9]"}

        block.execute(0.0, {0: 1.5}, params)
        first = params["_interp1d_"]

        params["interpolation"] = "nearest"
        result = block.execute(0.1, {0: 1.4}, params)[0]

        assert params["_interp1d_"] is not first
        np.testing.assert_allclose(result, [1.0])

    def test_2d_interpolator_is_reused_across_steps(self):
        from blocks.lookup_table import LookupTable2DBlock

        block = LookupTable2DBlock()
        params = {
            "x_values": "[0, 1, 2]",
            "y_values": "[0, 1]",
            "z_table": "[[0, 1], [2, 3], [4, 5]]",
        }

        block.execute(0.0, {0: 0.5, 1: 0.5}, params)
        first = params["_rgi_"]
        block.execute(0.1, {0: 1.5, 1: 0.5}, params)

        assert params["_rgi_"] is first

    def test_2d_interpolator_is_rebuilt_when_the_table_changes(self):
        from blocks.lookup_table import LookupTable2DBlock

        block = LookupTable2DBlock()
        params = {
            "x_values": "[0, 1, 2]",
            "y_values": "[0, 1]",
            "z_table": "[[0, 1], [2, 3], [4, 5]]",
        }

        first_out = block.execute(0.0, {0: 1.0, 1: 0.0}, params)[0]
        first = params["_rgi_"]

        params["z_table"] = "[[0, 10], [20, 30], [40, 50]]"
        second_out = block.execute(0.1, {0: 1.0, 1: 0.0}, params)[0]

        assert params["_rgi_"] is not first
        assert np.isclose(second_out, first_out * 10.0)
