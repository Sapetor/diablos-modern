"""Tests for VectorPerturb block."""

import pytest
import numpy as np
from blocks.optimization_primitives.vector_perturb import VectorPerturbBlock


class TestVectorPerturbBlock:
    """Tests for VectorPerturb block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = VectorPerturbBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "VectorPerturb"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.inputs) == 1
        assert len(self.block.outputs) == 1

    def test_perturb_first_component(self):
        """Test perturbing the first component."""
        params = {'index': 0, 'epsilon': 1e-6}
        x = np.array([1.0, 2.0, 3.0])

        result = self.block.execute(0.0, {0: x}, params)

        expected = np.array([1.0 + 1e-6, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result[0], expected)
        assert result['E'] is False

    def test_perturb_second_component(self):
        """Test perturbing the second component."""
        params = {'index': 1, 'epsilon': 0.01}
        x = np.array([1.0, 2.0, 3.0])

        result = self.block.execute(0.0, {0: x}, params)

        expected = np.array([1.0, 2.01, 3.0])
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_perturb_last_component(self):
        """Test perturbing the last component."""
        params = {'index': 2, 'epsilon': -0.5}  # Negative epsilon
        x = np.array([1.0, 2.0, 3.0])

        result = self.block.execute(0.0, {0: x}, params)

        expected = np.array([1.0, 2.0, 2.5])
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_does_not_modify_input(self):
        """Test that original input is not modified."""
        params = {'index': 0, 'epsilon': 1.0}
        x = np.array([1.0, 2.0])
        x_original = x.copy()

        self.block.execute(0.0, {0: x}, params)

        np.testing.assert_array_equal(x, x_original)

    def test_out_of_bounds_index(self):
        """Test behavior with out-of-bounds index."""
        params = {'index': 5, 'epsilon': 1.0}  # Index too large
        x = np.array([1.0, 2.0])

        result = self.block.execute(0.0, {0: x}, params)

        # Should return unchanged vector
        np.testing.assert_array_equal(result[0], x)

    def test_default_epsilon(self):
        """Test default epsilon value."""
        assert self.block.params['epsilon']['default'] == 1e-6
