"""Tests for the 1-D / 2-D Lookup Table blocks (blocks/lookup_table.py).

Instantiates each block and calls execute() with hand-chosen tables so the
interpolated/extrapolated outputs are exactly predictable.
"""

import numpy as np
import pytest

from blocks.lookup_table import LookupTable1DBlock, LookupTable2DBlock


def _p1(**ov):
    p = {"x_values": "[0, 1, 2, 3]", "y_values": "[0, 1, 4, 9]",
         "interpolation": "linear", "extrapolation": "clip"}
    p.update(ov)
    return p


def _p2(**ov):
    p = {"x_values": "[0, 1, 2]", "y_values": "[0, 1]",
         "z_table": "[[0, 1], [2, 3], [4, 5]]",
         "interpolation": "linear", "extrapolation": "clip"}
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
        assert np.isclose(below[0][0], 0.0)   # held at y[0]
        assert np.isclose(above[0][0], 9.0)   # held at y[-1]

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
            0.0, {0: np.array([1.5])},
            _p1(x_values="[3, 0, 2, 1]", y_values="[9, 0, 4, 1]"))
        assert np.isclose(out[0][0], 2.5)

    def test_length_mismatch_errors(self):
        out = self.b.execute(0.0, {0: 1.0},
                             _p1(x_values="[0, 1]", y_values="[0, 1, 2]"))
        assert out.get("E") is True

    def test_too_few_points_errors(self):
        out = self.b.execute(0.0, {0: 1.0}, _p1(x_values="[0]", y_values="[5]"))
        assert out.get("E") is True

    def test_duplicate_breakpoints_error(self):
        out = self.b.execute(0.0, {0: 1.0},
                             _p1(x_values="[0, 1, 1, 2]", y_values="[0, 1, 2, 3]"))
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
        out = self.b.execute(0.0, {0: 0.0, 1: 0.0},
                             _p2(x_values="[0]", z_table="[[0, 1]]"))
        assert out.get("E") is True
