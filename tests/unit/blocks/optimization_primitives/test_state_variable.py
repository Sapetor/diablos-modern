"""Tests for StateVariable block."""

import pytest
import numpy as np
from blocks.optimization_primitives.state_variable import StateVariableBlock


class TestStateVariableBlock:
    """Tests for StateVariable block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = StateVariableBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "StateVariable"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.inputs) == 1
        assert len(self.block.outputs) == 1
        assert self.block.requires_inputs is False

    def test_initial_value(self):
        """Test that initial value is output on first call."""
        params = {
            'initial_value': [5.0, 3.0],
            'dimension': 2,
            '_initialized_': False
        }

        result = self.block.execute(0.0, {}, params)

        np.testing.assert_array_equal(result[0], [5.0, 3.0])
        assert result['E'] is False

    def test_state_update(self):
        """Test that state is updated from input."""
        params = {
            'initial_value': [1.0, 1.0],
            'dimension': 2,
            '_initialized_': False
        }

        # First call - returns initial
        result1 = self.block.execute(0.0, {}, params)
        np.testing.assert_array_equal(result1[0], [1.0, 1.0])

        # Second call with new input - still returns old state (current iteration)
        result2 = self.block.execute(0.1, {0: np.array([2.0, 3.0])}, params)
        np.testing.assert_array_equal(result2[0], [1.0, 1.0])

        # Third call - now returns the updated state
        result3 = self.block.execute(0.2, {}, params)
        np.testing.assert_array_equal(result3[0], [2.0, 3.0])

    def test_iteration_sequence(self):
        """Test a sequence of iterations like gradient descent."""
        params = {
            'initial_value': [10.0, 10.0],
            'dimension': 2,
            '_initialized_': False
        }

        # Iteration 0: x = [10, 10]
        x0 = self.block.execute(0.0, {}, params)[0]
        np.testing.assert_array_equal(x0, [10.0, 10.0])

        # Compute update: x_next = x - 0.1 * gradient (simulated)
        x_next = x0 - 0.1 * np.array([20.0, 20.0])  # gradient = 2x
        self.block.execute(0.1, {0: x_next}, params)

        # Iteration 1: x = [8, 8]
        x1 = self.block.execute(0.1, {}, params)[0]
        np.testing.assert_array_equal(x1, [8.0, 8.0])

    def test_string_initial_value(self):
        """Test that string initial value is parsed correctly."""
        params = {
            'initial_value': '[1.0, 2.0, 3.0]',
            'dimension': 3,
            '_initialized_': False
        }

        result = self.block.execute(0.0, {}, params)

        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    def test_single_variable(self):
        """Test with single state variable."""
        params = {
            'initial_value': [5.0],
            'dimension': 1,
            '_initialized_': False
        }

        result = self.block.execute(0.0, {}, params)

        np.testing.assert_array_equal(result[0], [5.0])
