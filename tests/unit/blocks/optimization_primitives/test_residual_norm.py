"""Tests for ResidualNorm block."""

import pytest
import numpy as np
from blocks.optimization_primitives.residual_norm import ResidualNormBlock


class TestResidualNormBlock:
    """Tests for ResidualNorm block."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = ResidualNormBlock()

    def test_block_properties(self):
        """Test block metadata."""
        assert self.block.block_name == "ResidualNorm"
        assert self.block.category == "Optimization Primitives"
        assert len(self.block.inputs) == 1
        assert len(self.block.outputs) == 1

    def test_l2_norm(self):
        """Test L2 (Euclidean) norm."""
        params = {'norm_type': '2'}
        F = np.array([3.0, 4.0])

        result = self.block.execute(0.0, {0: F}, params)

        assert result[0] == pytest.approx(5.0)
        assert result['E'] is False

    def test_l1_norm(self):
        """Test L1 (Manhattan) norm."""
        params = {'norm_type': '1'}
        F = np.array([3.0, -4.0])

        result = self.block.execute(0.0, {0: F}, params)

        assert result[0] == pytest.approx(7.0)

    def test_inf_norm(self):
        """Test infinity (max) norm."""
        params = {'norm_type': 'inf'}
        F = np.array([3.0, -7.0, 2.0])

        result = self.block.execute(0.0, {0: F}, params)

        assert result[0] == pytest.approx(7.0)

    def test_zero_vector(self):
        """Test norm of zero vector (convergence check)."""
        params = {'norm_type': '2'}
        F = np.array([0.0, 0.0, 0.0])

        result = self.block.execute(0.0, {0: F}, params)

        assert result[0] == pytest.approx(0.0)

    def test_single_element(self):
        """Test norm of single element."""
        params = {'norm_type': '2'}
        F = np.array([-5.0])

        result = self.block.execute(0.0, {0: F}, params)

        assert result[0] == pytest.approx(5.0)

    def test_default_is_l2(self):
        """Test default norm type is L2."""
        assert self.block.params['norm_type']['default'] == '2'

    def test_convergence_monitoring(self):
        """Test typical use case: monitoring gradient norm convergence."""
        params = {'norm_type': '2'}

        # Simulate decreasing gradient norms
        gradients = [
            np.array([10.0, 10.0]),
            np.array([5.0, 5.0]),
            np.array([1.0, 1.0]),
            np.array([0.1, 0.1]),
        ]

        norms = []
        for grad in gradients:
            result = self.block.execute(0.0, {0: grad}, params)
            norms.append(result[0])

        # Verify norms are decreasing
        for i in range(len(norms) - 1):
            assert norms[i] > norms[i + 1]
