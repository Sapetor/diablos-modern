"""Tests for RootFinder block."""

import pytest
import numpy as np
from blocks.optimization_primitives.root_finder import RootFinderBlock


class TestRootFinderBlock:
    """Tests for RootFinder block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = RootFinderBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "RootFinder"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.inputs) == 1
        assert len(self.block.outputs) == 1

    def test_linear_system(self):
        """Test Newton step on linear system (should solve in one step)."""
        # F(x) = x1 - 2, x2 - 3
        # Solution: x = [2, 3]
        params = {
            'expressions': 'x1 - 2, x2 - 3',
            'variables': 'x1,x2',
            'epsilon': 1e-6,
            'damping': 1.0
        }

        x = np.array([0.0, 0.0])
        result = self.block.execute(0.0, {0: x}, params)

        np.testing.assert_array_almost_equal(result[0], [2.0, 3.0])
        assert result['E'] is False

    def test_quadratic_system(self):
        """Test Newton step on quadratic system."""
        # F(x) = x1**2 - 1
        # Starting at x=2, F(2)=3, F'(2)=4
        # Newton step: x_new = 2 - 3/4 = 1.25
        params = {
            'expressions': 'x1**2 - 1',
            'variables': 'x1',
            'epsilon': 1e-6,
            'damping': 1.0
        }

        x = np.array([2.0])
        result = self.block.execute(0.0, {0: x}, params)

        assert result[0][0] == pytest.approx(1.25, rel=1e-4)

    def test_damping(self):
        """Test damped Newton step."""
        params = {
            'expressions': 'x1 - 2',
            'variables': 'x1',
            'epsilon': 1e-6,
            'damping': 0.5  # Half step
        }

        x = np.array([0.0])
        result = self.block.execute(0.0, {0: x}, params)

        # Full step would give 2, half step gives 1
        assert result[0][0] == pytest.approx(1.0)

    def test_circle_intersection(self):
        """Test system of two circles (classic example)."""
        # x1**2 + x2 - 1 = 0
        # x1 + x2**2 - 1 = 0
        # Solutions include (0, 1), (1, 0)
        params = {
            'expressions': 'x1**2 + x2 - 1, x1 + x2**2 - 1',
            'variables': 'x1,x2',
            'epsilon': 1e-6,
            'damping': 1.0
        }

        # Start near (1, 0)
        x = np.array([0.8, 0.2])

        # Run several iterations
        for _ in range(10):
            result = self.block.execute(0.0, {0: x}, params)
            x = result[0]

        # Should converge to (1, 0) or (0, 1)
        # Check if we're close to either solution
        dist_to_10 = np.linalg.norm(x - np.array([1.0, 0.0]))
        dist_to_01 = np.linalg.norm(x - np.array([0.0, 1.0]))

        assert min(dist_to_10, dist_to_01) < 0.01

    def test_convergence_from_far(self):
        """Test convergence from a far starting point."""
        # Simple 1D: x - 5 = 0
        params = {
            'expressions': 'x1 - 5',
            'variables': 'x1',
            'epsilon': 1e-6,
            'damping': 1.0
        }

        x = np.array([100.0])  # Far from solution
        result = self.block.execute(0.0, {0: x}, params)

        # Linear system should solve in one step
        assert result[0][0] == pytest.approx(5.0)
