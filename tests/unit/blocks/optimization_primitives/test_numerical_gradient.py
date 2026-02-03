"""Tests for NumericalGradient block."""

import pytest
import numpy as np
from blocks.optimization_primitives.numerical_gradient import NumericalGradientBlock


class TestNumericalGradientBlock:
    """Tests for NumericalGradient block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = NumericalGradientBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "NumericalGradient"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.outputs) == 1

    def test_forward_difference_2d(self):
        """Test forward difference gradient for 2D function."""
        params = {'dimension': 2, 'epsilon': 1e-6, 'method': 'forward'}

        # For f(x) = x1**2 + x2**2 at [3, 4]:
        # f(x) = 25
        # f(x + eps*e1) = (3+eps)**2 + 16 ≈ 25 + 6*eps
        # f(x + eps*e2) = 9 + (4+eps)**2 ≈ 25 + 8*eps
        # grad = [6, 8]

        eps = 1e-6
        f_center = 25.0
        f_plus_0 = 25.0 + 6 * eps  # ∂f/∂x1 * eps
        f_plus_1 = 25.0 + 8 * eps  # ∂f/∂x2 * eps

        inputs = {
            0: f_center,
            1: f_plus_0,
            2: f_plus_1
        }

        result = self.block.execute(0.0, inputs, params)

        np.testing.assert_array_almost_equal(result[0], [6.0, 8.0], decimal=5)
        assert result['E'] is False

    def test_central_difference_2d(self):
        """Test central difference gradient for 2D function."""
        params = {'dimension': 2, 'epsilon': 1e-6, 'method': 'central'}

        eps = 1e-6
        # For f(x) = x**2 at x=3: f'=6
        # Central: (f(x+eps) - f(x-eps)) / 2eps = ((3+eps)**2 - (3-eps)**2) / 2eps
        # = (12*eps) / 2eps = 6

        inputs = {
            0: 25.0,  # f_center (not used in central)
            1: 25.0 + 6 * eps,  # f_plus_0
            2: 25.0 + 8 * eps,  # f_plus_1
            3: 25.0 - 6 * eps,  # f_minus_0
            4: 25.0 - 8 * eps,  # f_minus_1
        }

        result = self.block.execute(0.0, inputs, params)

        np.testing.assert_array_almost_equal(result[0], [6.0, 8.0], decimal=5)

    def test_dynamic_inputs_forward(self):
        """Test that inputs are dynamically generated based on dimension."""
        params = {'dimension': 3, 'epsilon': 1e-6, 'method': 'forward'}
        inputs = self.block.get_inputs(params)

        # Should have f_center + 3 f_plus inputs
        assert len(inputs) == 4
        assert inputs[0]['name'] == 'f_center'
        assert inputs[1]['name'] == 'f_plus_0'
        assert inputs[2]['name'] == 'f_plus_1'
        assert inputs[3]['name'] == 'f_plus_2'

    def test_dynamic_inputs_central(self):
        """Test inputs for central difference."""
        params = {'dimension': 2, 'epsilon': 1e-6, 'method': 'central'}
        inputs = self.block.get_inputs(params)

        # Should have f_center + 2 f_plus + 2 f_minus
        assert len(inputs) == 5
        assert inputs[3]['name'] == 'f_minus_0'
        assert inputs[4]['name'] == 'f_minus_1'

    def test_zero_gradient(self):
        """Test gradient at minimum (should be zero)."""
        params = {'dimension': 2, 'epsilon': 1e-6, 'method': 'forward'}

        # At minimum, all f values are the same
        f = 0.0
        inputs = {0: f, 1: f, 2: f}

        result = self.block.execute(0.0, inputs, params)

        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])
