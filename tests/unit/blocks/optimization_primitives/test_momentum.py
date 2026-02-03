"""Tests for Momentum block."""

import pytest
import numpy as np
from blocks.optimization_primitives.momentum import MomentumBlock


class TestMomentumBlock:
    """Tests for Momentum block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = MomentumBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "Momentum"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.inputs) == 1
        assert len(self.block.outputs) == 1

    def test_first_step(self):
        """Test first momentum step (velocity starts at zero)."""
        params = {
            'alpha': 0.1,
            'beta': 0.9,
            '_initialized_': False
        }
        grad = np.array([2.0, 4.0])

        result = self.block.execute(0.0, {0: grad}, params)

        # v = 0.9*0 - 0.1*[2,4] = [-0.2, -0.4]
        np.testing.assert_array_almost_equal(result[0], [-0.2, -0.4])
        assert result['E'] is False

    def test_velocity_accumulation(self):
        """Test that velocity accumulates across iterations."""
        params = {
            'alpha': 0.1,
            'beta': 0.9,
            '_initialized_': False
        }
        grad = np.array([2.0, 2.0])

        # First step: v = -0.1 * [2,2] = [-0.2, -0.2]
        v1 = self.block.execute(0.0, {0: grad}, params)[0]
        np.testing.assert_array_almost_equal(v1, [-0.2, -0.2])

        # Second step: v = 0.9*[-0.2,-0.2] - 0.1*[2,2] = [-0.18-0.2, -0.18-0.2] = [-0.38, -0.38]
        v2 = self.block.execute(0.1, {0: grad}, params)[0]
        np.testing.assert_array_almost_equal(v2, [-0.38, -0.38])

        # Third step: v = 0.9*[-0.38,-0.38] - 0.1*[2,2] = [-0.342-0.2, ...] = [-0.542, -0.542]
        v3 = self.block.execute(0.2, {0: grad}, params)[0]
        np.testing.assert_array_almost_equal(v3, [-0.542, -0.542])

    def test_zero_momentum(self):
        """Test with zero momentum (beta=0) - should be like standard GD."""
        params = {
            'alpha': 0.1,
            'beta': 0.0,  # No momentum
            '_initialized_': False
        }
        grad = np.array([5.0, 10.0])

        result = self.block.execute(0.0, {0: grad}, params)

        # v = 0*0 - 0.1*[5,10] = [-0.5, -1.0]
        np.testing.assert_array_almost_equal(result[0], [-0.5, -1.0])

        # Second step should be the same (no accumulation)
        result2 = self.block.execute(0.1, {0: grad}, params)
        np.testing.assert_array_almost_equal(result2[0], [-0.5, -1.0])

    def test_high_momentum(self):
        """Test with high momentum coefficient."""
        params = {
            'alpha': 0.01,
            'beta': 0.99,  # High momentum
            '_initialized_': False
        }
        grad = np.array([1.0, 1.0])

        # Run many iterations to see velocity grow
        for i in range(500):  # Need more iterations for high momentum to converge
            result = self.block.execute(i * 0.1, {0: grad}, params)

        # With constant gradient and high momentum, velocity should approach -alpha*grad/(1-beta)
        # = -0.01 * [1,1] / 0.01 = [-1, -1]
        v_final = result[0]
        np.testing.assert_array_almost_equal(v_final, [-1.0, -1.0], decimal=1)

    def test_default_parameters(self):
        """Test default parameters."""
        assert self.block.params['alpha']['default'] == 0.01
        assert self.block.params['beta']['default'] == 0.9

    def test_zero_gradient(self):
        """Test with zero gradient."""
        params = {
            'alpha': 0.1,
            'beta': 0.9,
            '_initialized_': False
        }
        grad = np.array([0.0, 0.0])

        result = self.block.execute(0.0, {0: grad}, params)

        np.testing.assert_array_equal(result[0], [0.0, 0.0])
