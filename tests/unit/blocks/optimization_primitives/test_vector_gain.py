"""Tests for VectorGain block."""

import pytest
import numpy as np
from blocks.optimization_primitives.vector_gain import VectorGainBlock


class TestVectorGainBlock:
    """Tests for VectorGain block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = VectorGainBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "VectorGain"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.inputs) == 1
        assert len(self.block.outputs) == 1

    def test_positive_gain(self):
        """Test scaling with positive gain."""
        params = {'gain': 2.0}
        x = np.array([1.0, 2.0, 3.0])

        result = self.block.execute(0.0, {0: x}, params)

        np.testing.assert_array_equal(result[0], [2.0, 4.0, 6.0])
        assert result['E'] is False

    def test_negative_gain(self):
        """Test scaling with negative gain (for gradient descent)."""
        params = {'gain': -0.1}
        grad = np.array([6.0, 8.0])

        result = self.block.execute(0.0, {0: grad}, params)

        np.testing.assert_array_almost_equal(result[0], [-0.6, -0.8])

    def test_zero_gain(self):
        """Test scaling with zero gain."""
        params = {'gain': 0.0}
        x = np.array([1.0, 2.0, 3.0])

        result = self.block.execute(0.0, {0: x}, params)

        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0])

    def test_fractional_gain(self):
        """Test scaling with fractional gain (learning rate)."""
        params = {'gain': 0.01}
        grad = np.array([100.0, 200.0])

        result = self.block.execute(0.0, {0: grad}, params)

        np.testing.assert_array_equal(result[0], [1.0, 2.0])

    def test_single_element(self):
        """Test with single element vector."""
        params = {'gain': 5.0}
        x = np.array([3.0])

        result = self.block.execute(0.0, {0: x}, params)

        np.testing.assert_array_equal(result[0], [15.0])

    def test_default_gain(self):
        """Test default gain is 1.0 (identity)."""
        assert self.block.params['gain']['default'] == 1.0
