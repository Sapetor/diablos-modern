"""Tests for ObjectiveFunction block."""

import pytest
import numpy as np
from blocks.optimization_primitives.objective_function import ObjectiveFunctionBlock


class TestObjectiveFunctionBlock:
    """Tests for ObjectiveFunction block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = ObjectiveFunctionBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "ObjectiveFunction"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.inputs) == 1
        assert len(self.block.outputs) == 1

    def test_quadratic_function(self):
        """Test evaluation of x1**2 + x2**2."""
        params = {
            'expression': 'x1**2 + x2**2',
            'variables': 'x1,x2'
        }

        # Test at origin
        result = self.block.execute(0.0, {0: np.array([0.0, 0.0])}, params)
        assert result[0] == pytest.approx(0.0)
        assert result['E'] is False

        # Test at [1, 1]
        result = self.block.execute(0.0, {0: np.array([1.0, 1.0])}, params)
        assert result[0] == pytest.approx(2.0)

        # Test at [3, 4]
        result = self.block.execute(0.0, {0: np.array([3.0, 4.0])}, params)
        assert result[0] == pytest.approx(25.0)

    def test_rosenbrock_function(self):
        """Test Rosenbrock function: (1-x1)**2 + 100*(x2-x1**2)**2."""
        params = {
            'expression': '(1-x1)**2 + 100*(x2-x1**2)**2',
            'variables': 'x1,x2'
        }

        # At minimum [1, 1], f should be 0
        result = self.block.execute(0.0, {0: np.array([1.0, 1.0])}, params)
        assert result[0] == pytest.approx(0.0)

        # At [0, 0], f = 1 + 0 = 1
        result = self.block.execute(0.0, {0: np.array([0.0, 0.0])}, params)
        assert result[0] == pytest.approx(1.0)

    def test_math_functions(self):
        """Test that math functions (sin, cos, exp, etc.) work."""
        params = {
            'expression': 'sin(x1) + cos(x2)',
            'variables': 'x1,x2'
        }

        result = self.block.execute(0.0, {0: np.array([0.0, 0.0])}, params)
        # sin(0) + cos(0) = 0 + 1 = 1
        assert result[0] == pytest.approx(1.0)

        result = self.block.execute(0.0, {0: np.array([np.pi/2, 0.0])}, params)
        # sin(pi/2) + cos(0) = 1 + 1 = 2
        assert result[0] == pytest.approx(2.0)

    def test_single_variable(self):
        """Test single variable function."""
        params = {
            'expression': 'x1**3 - 2*x1 + 1',
            'variables': 'x1'
        }

        result = self.block.execute(0.0, {0: np.array([2.0])}, params)
        # 8 - 4 + 1 = 5
        assert result[0] == pytest.approx(5.0)

    def test_three_variables(self):
        """Test function with 3 variables."""
        params = {
            'expression': 'x1 + 2*x2 + 3*x3',
            'variables': 'x1,x2,x3'
        }

        result = self.block.execute(0.0, {0: np.array([1.0, 2.0, 3.0])}, params)
        # 1 + 4 + 9 = 14
        assert result[0] == pytest.approx(14.0)

    def test_default_expression(self):
        """Test default expression works."""
        params = self.block.params
        # Extract defaults
        defaults = {k: v['default'] for k, v in params.items()}

        result = self.block.execute(0.0, {0: np.array([2.0, 3.0])}, defaults)
        # x1**2 + x2**2 = 4 + 9 = 13
        assert result[0] == pytest.approx(13.0)
