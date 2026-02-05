"""Tests for Product block."""

import pytest
import numpy as np
from blocks.product import ProductBlock


class TestProductBlock:
    """Tests for Product block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = ProductBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "Product"
        assert self.block.category == "Math"
        assert len(self.block.inputs) == 2
        assert len(self.block.outputs) == 1

    def test_default_multiplication(self):
        """Test default operation is multiplication."""
        result = self.block.execute(0.0, {0: 4.0, 1: 3.0}, {'ops': '**'})
        assert np.isclose(result[0][0], 12.0)

    def test_division(self):
        """Test division operation."""
        result = self.block.execute(0.0, {0: 12.0, 1: 4.0}, {'ops': '*/'})
        assert np.isclose(result[0][0], 3.0)

    def test_multiply_then_divide(self):
        """Test multiply first, then divide."""
        # ops='*/' means: result = input0 / input1
        # First char is for first input (multiply by it), second for second input
        result = self.block.execute(0.0, {0: 6.0, 1: 2.0}, {'ops': '*/'})
        assert np.isclose(result[0][0], 3.0)

    def test_three_inputs(self):
        """Test with three inputs."""
        result = self.block.execute(0.0, {0: 2.0, 1: 3.0, 2: 4.0}, {'ops': '***'})
        assert np.isclose(result[0][0], 24.0)

    def test_three_inputs_with_division(self):
        """Test with three inputs including division."""
        # 2 * 6 / 3 = 4
        result = self.block.execute(0.0, {0: 2.0, 1: 6.0, 2: 3.0}, {'ops': '**/'})
        assert np.isclose(result[0][0], 4.0)

    def test_array_inputs(self):
        """Test with array inputs."""
        result = self.block.execute(0.0, {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}, {'ops': '**'})
        np.testing.assert_array_almost_equal(result[0], [3.0, 8.0])

    def test_division_by_zero_handling(self):
        """Test division by zero returns finite value."""
        result = self.block.execute(0.0, {0: 1.0, 1: 0.0}, {'ops': '*/'})
        # Should return large finite number, not inf
        assert np.isfinite(result[0][0])

    def test_newton_step_calculation(self):
        """Test Newton's method step: f(x)/f'(x) for f(x) = 4x³."""
        # At x=3: f(3) = 4*27 = 108, f'(3) = 12*9 = 108
        # Newton step = f/f' = 1
        result = self.block.execute(0.0, {0: 108.0, 1: 108.0}, {'ops': '*/'})
        assert np.isclose(result[0][0], 1.0)

    def test_get_inputs_dynamic(self):
        """Test that get_inputs returns correct number based on ops."""
        inputs_2 = self.block.get_inputs({'ops': '**'})
        assert len(inputs_2) == 2

        inputs_3 = self.block.get_inputs({'ops': '***'})
        assert len(inputs_3) == 3

        inputs_4 = self.block.get_inputs({'ops': '*/**'})
        assert len(inputs_4) == 4

    def test_symbolic_execute(self):
        """Test symbolic execution."""
        try:
            from sympy import Symbol
            u0 = Symbol('u0')
            u1 = Symbol('u1')

            result = self.block.symbolic_execute({0: u0, 1: u1}, {'ops': '*/'})
            # Result should be u0 / u1
            assert result is not None
            assert 0 in result
        except ImportError:
            pytest.skip("sympy not available")
