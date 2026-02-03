"""Tests for VectorSum block."""

import pytest
import numpy as np
from blocks.optimization_primitives.vector_sum import VectorSumBlock


class TestVectorSumBlock:
    """Tests for VectorSum block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = VectorSumBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "VectorSum"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.outputs) == 1

    def test_addition(self):
        """Test vector addition."""
        params = {'signs': '++'}
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])

        result = self.block.execute(0.0, {0: x1, 1: x2}, params)

        np.testing.assert_array_equal(result[0], [4.0, 6.0])
        assert result['E'] is False

    def test_subtraction(self):
        """Test vector subtraction (gradient descent update)."""
        params = {'signs': '+-'}
        x = np.array([10.0, 10.0])
        update = np.array([1.0, 2.0])

        result = self.block.execute(0.0, {0: x, 1: update}, params)

        np.testing.assert_array_equal(result[0], [9.0, 8.0])

    def test_three_inputs(self):
        """Test with three inputs."""
        params = {'signs': '++-'}
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])
        x3 = np.array([0.5, 0.5])

        result = self.block.execute(0.0, {0: x1, 1: x2, 2: x3}, params)

        # 1+3-0.5=3.5, 2+4-0.5=5.5
        np.testing.assert_array_equal(result[0], [3.5, 5.5])

    def test_negative_first(self):
        """Test starting with negative sign."""
        params = {'signs': '-+'}
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])

        result = self.block.execute(0.0, {0: x1, 1: x2}, params)

        # -1+3=2, -2+4=2
        np.testing.assert_array_equal(result[0], [2.0, 2.0])

    def test_dynamic_inputs(self):
        """Test that inputs are dynamically generated based on signs."""
        params = {'signs': '+++'}
        inputs = self.block.get_inputs(params)

        assert len(inputs) == 3
        assert inputs[0]['name'] == 'x1'
        assert inputs[1]['name'] == 'x2'
        assert inputs[2]['name'] == 'x3'

    def test_single_input(self):
        """Test with single input (passthrough)."""
        params = {'signs': '+'}
        x = np.array([1.0, 2.0, 3.0])

        result = self.block.execute(0.0, {0: x}, params)

        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    def test_gradient_descent_step(self):
        """Test typical gradient descent step: x_new = x - alpha*grad."""
        params = {'signs': '+-'}
        x = np.array([5.0, 5.0])
        alpha_grad = np.array([1.0, 1.0])  # alpha * gradient

        result = self.block.execute(0.0, {0: x, 1: alpha_grad}, params)

        np.testing.assert_array_equal(result[0], [4.0, 4.0])
