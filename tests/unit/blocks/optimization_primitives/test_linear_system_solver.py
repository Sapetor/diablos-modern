"""Tests for LinearSystemSolver block."""

import pytest
import numpy as np
from blocks.optimization_primitives.linear_system_solver import LinearSystemSolverBlock


class TestLinearSystemSolverBlock:
    """Tests for LinearSystemSolver block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = LinearSystemSolverBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "LinearSystemSolver"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.inputs) == 2
        assert len(self.block.outputs) == 1

    def test_simple_2x2_system(self):
        """Test solving a simple 2x2 system."""
        # Ax = b where A = [[2, 0], [0, 3]], b = [4, 9]
        # Solution: x = [2, 3]
        params = {'method': 'direct', 'dimension': 2, 'regularization': 0.0}
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([4.0, 9.0])

        result = self.block.execute(0.0, {0: A, 1: b}, params)

        np.testing.assert_array_almost_equal(result[0], [2.0, 3.0])
        assert result['E'] is False

    def test_coupled_system(self):
        """Test solving a coupled 2x2 system."""
        # A = [[1, 1], [2, -1]], b = [3, 3]
        # Solution: x = [2, 1]
        params = {'method': 'direct', 'dimension': 2, 'regularization': 0.0}
        A = np.array([[1.0, 1.0], [2.0, -1.0]])
        b = np.array([3.0, 3.0])

        result = self.block.execute(0.0, {0: A, 1: b}, params)

        np.testing.assert_array_almost_equal(result[0], [2.0, 1.0])

    def test_lstsq_method(self):
        """Test least squares method."""
        params = {'method': 'lstsq', 'dimension': 2, 'regularization': 0.0}
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([5.0, 7.0])

        result = self.block.execute(0.0, {0: A, 1: b}, params)

        np.testing.assert_array_almost_equal(result[0], [5.0, 7.0])

    def test_pinv_method(self):
        """Test pseudo-inverse method."""
        params = {'method': 'pinv', 'dimension': 2, 'regularization': 0.0}
        A = np.array([[1.0, 0.0], [0.0, 2.0]])
        b = np.array([3.0, 8.0])

        result = self.block.execute(0.0, {0: A, 1: b}, params)

        np.testing.assert_array_almost_equal(result[0], [3.0, 4.0])

    def test_flattened_matrix(self):
        """Test with flattened matrix input."""
        params = {'method': 'direct', 'dimension': 2, 'regularization': 0.0}
        A_flat = np.array([1.0, 0.0, 0.0, 1.0])  # Identity matrix flattened
        b = np.array([5.0, 3.0])

        result = self.block.execute(0.0, {0: A_flat, 1: b}, params)

        np.testing.assert_array_almost_equal(result[0], [5.0, 3.0])

    def test_regularization(self):
        """Test Tikhonov regularization."""
        # Near-singular matrix, regularization helps
        params = {'method': 'direct', 'dimension': 2, 'regularization': 0.1}
        A = np.array([[1.0, 1.0], [1.0, 1.0]])  # Singular
        b = np.array([2.0, 2.0])

        result = self.block.execute(0.0, {0: A, 1: b}, params)

        # With regularization, should get a solution
        assert result['E'] is False
        assert len(result[0]) == 2

    def test_3x3_system(self):
        """Test 3x3 system."""
        params = {'method': 'direct', 'dimension': 3, 'regularization': 0.0}
        A = np.eye(3) * 2  # 2*I
        b = np.array([4.0, 6.0, 8.0])

        result = self.block.execute(0.0, {0: A, 1: b}, params)

        np.testing.assert_array_almost_equal(result[0], [2.0, 3.0, 4.0])
