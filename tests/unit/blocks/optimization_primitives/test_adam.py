"""Tests for Adam block."""

import pytest
import numpy as np
from blocks.optimization_primitives.adam import AdamBlock


class TestAdamBlock:
    """Tests for Adam block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = AdamBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "Adam"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.inputs) == 1
        assert len(self.block.outputs) == 1

    def test_first_step(self):
        """Test first Adam step."""
        params = {
            'alpha': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            '_initialized_': False
        }
        grad = np.array([1.0, 1.0])

        result = self.block.execute(0.0, {0: grad}, params)

        # First step calculation:
        # m = 0.9*0 + 0.1*[1,1] = [0.1, 0.1]
        # v = 0.999*0 + 0.001*[1,1] = [0.001, 0.001]
        # m_hat = [0.1, 0.1] / (1-0.9) = [1, 1]
        # v_hat = [0.001, 0.001] / (1-0.999) = [1, 1]
        # update = -0.001 * [1,1] / (sqrt([1,1]) + 1e-8) ≈ [-0.001, -0.001]

        expected = -0.001 * np.ones(2)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)
        assert result['E'] is False

    def test_update_direction(self):
        """Test that update is in opposite direction of gradient."""
        params = {
            'alpha': 0.1,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            '_initialized_': False
        }
        grad = np.array([5.0, -3.0])

        result = self.block.execute(0.0, {0: grad}, params)

        # Update should be opposite sign of gradient
        assert result[0][0] < 0  # grad[0] > 0, update[0] < 0
        assert result[0][1] > 0  # grad[1] < 0, update[1] > 0

    def test_adaptive_learning(self):
        """Test that Adam adapts to gradient magnitude."""
        params = {
            'alpha': 0.1,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            '_initialized_': False
        }

        # Large gradient - should get scaled down
        large_grad = np.array([100.0, 100.0])
        result1 = self.block.execute(0.0, {0: large_grad}, params)

        # Reset
        params['_initialized_'] = False

        # Small gradient - relative update should be similar
        small_grad = np.array([1.0, 1.0])
        result2 = self.block.execute(0.0, {0: small_grad}, params)

        # Due to bias correction in first step, updates should be similar magnitude
        ratio = np.abs(result1[0][0] / result2[0][0])
        # Ratio should be much less than 100 (the gradient ratio)
        assert ratio < 50

    def test_bias_correction(self):
        """Test that bias correction works properly."""
        params = {
            'alpha': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            '_initialized_': False
        }
        grad = np.array([1.0])

        # First few steps should have larger updates due to bias correction
        updates = []
        for i in range(10):
            result = self.block.execute(i * 0.1, {0: grad}, params)
            updates.append(np.abs(result[0][0]))

        # Updates should start larger and decrease as bias correction diminishes
        assert updates[0] > updates[5]

    def test_default_parameters(self):
        """Test default parameters match standard Adam."""
        assert self.block.params['alpha']['default'] == 0.001
        assert self.block.params['beta1']['default'] == 0.9
        assert self.block.params['beta2']['default'] == 0.999
        assert self.block.params['epsilon']['default'] == 1e-8

    def test_zero_gradient(self):
        """Test with zero gradient."""
        params = {
            'alpha': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            '_initialized_': False
        }
        grad = np.array([0.0, 0.0])

        result = self.block.execute(0.0, {0: grad}, params)

        np.testing.assert_array_equal(result[0], [0.0, 0.0])

    def test_multiple_iterations(self):
        """Test convergence behavior over multiple iterations."""
        params = {
            'alpha': 0.1,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            '_initialized_': False
        }

        # Constant gradient - simulate optimization
        grad = np.array([2.0, 2.0])

        for i in range(100):
            result = self.block.execute(i * 0.1, {0: grad}, params)

        # After many iterations, should reach steady state
        # Updates should be non-zero and in correct direction
        assert result[0][0] < 0
        assert result[0][1] < 0
