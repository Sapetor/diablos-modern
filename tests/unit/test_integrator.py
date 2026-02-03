"""
Unit tests for Integrator block.
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestIntegratorBlock:
    """Tests for Integrator block."""

    def test_integrator_initial_condition(self):
        """Test integrator outputs initial condition at t=0."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        params = {'init_conds': 5.0, 'method': 'FWD_EULER', '_init_start_': True}

        # First call with output_only to get initial value
        result = block.execute(time=0.0, inputs={0: np.array([0.0])}, params=params, output_only=True)
        assert np.isclose(result[0][0], 5.0), "Initial output should equal init_conds"

    def test_integrator_fwd_euler(self):
        """Test forward Euler integration accumulates correctly."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        dtime = 0.1
        params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        # First call initializes and integrates: 0 + 0.1*1 = 0.1
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=dtime)
        assert np.isclose(result[0][0], 0.1, atol=0.01), f"FWD_EULER first step: expected ~0.1, got {result[0][0]}"

    def test_integrator_fwd_euler_accumulation(self):
        """Test forward Euler accumulates over multiple steps."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        dtime = 0.1
        params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        # Run 10 steps with constant input of 1.0
        for i in range(10):
            result = block.execute(time=i*dtime, inputs={0: np.array([1.0])}, params=params, dtime=dtime)

        # After 10 steps: 10 * 0.1 * 1.0 = 1.0
        assert np.isclose(result[0][0], 1.0, atol=0.1), f"Accumulated integral should be ~1.0, got {result[0][0]}"

    def test_integrator_solve_ivp(self):
        """Test SOLVE_IVP integration method."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        dtime = 0.1
        params = {'init_conds': 0.0, 'method': 'SOLVE_IVP', '_init_start_': True, '_name_': 'TestInt'}

        # First call initializes and integrates
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=dtime)
        # After one step integrating constant 1.0 over dtime=0.1, should be ~0.1
        assert np.isclose(result[0][0], 0.1, atol=0.02), f"SOLVE_IVP: expected ~0.1, got {result[0][0]}"

    def test_integrator_solve_ivp_accumulation(self):
        """Test SOLVE_IVP accumulates over multiple steps."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        dtime = 0.1
        params = {'init_conds': 0.0, 'method': 'SOLVE_IVP', '_init_start_': True, '_name_': 'TestInt'}

        # Run 10 steps with constant input of 1.0
        for i in range(10):
            result = block.execute(time=i*dtime, inputs={0: np.array([1.0])}, params=params, dtime=dtime)

        # After 10 steps: integral of 1.0 over 1.0s should be ~1.0
        assert np.isclose(result[0][0], 1.0, atol=0.1), f"SOLVE_IVP accumulated: expected ~1.0, got {result[0][0]}"

    def test_integrator_vector_input(self):
        """Test integrator handles vector inputs."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        dtime = 0.1
        params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        vec_input = np.array([1.0, 2.0, 3.0])

        # Execute - will expand init_conds to match input dimensions
        result = block.execute(time=0.0, inputs={0: vec_input}, params=params, dtime=dtime)

        assert result[0].shape == vec_input.shape, "Output shape should match input"
        # After one step: dtime * [1, 2, 3] = [0.1, 0.2, 0.3]
        expected = dtime * vec_input
        assert np.allclose(result[0], expected, atol=0.01), f"Expected {expected}, got {result[0]}"

    def test_integrator_tustin(self):
        """Test Tustin (trapezoidal) integration."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        dtime = 0.1
        params = {'init_conds': 0.0, 'method': 'TUSTIN', '_init_start_': True, '_name_': 'TestInt'}

        # First call initializes with mem_list containing zeros
        # Tustin: x_new = x_old + 0.5 * dtime * (u_old + u_new)
        # First step: u_old from mem_list = 0, u_new = 1.0
        # x = 0 + 0.5 * 0.1 * (0 + 1) = 0.05
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=dtime)

        # Second step: u_old = 1.0, u_new = 1.0
        # x = 0.05 + 0.5 * 0.1 * (1 + 1) = 0.05 + 0.1 = 0.15
        result = block.execute(time=dtime, inputs={0: np.array([1.0])}, params=params, dtime=dtime)
        assert np.isclose(result[0][0], 0.15, atol=0.02), f"TUSTIN: expected ~0.15, got {result[0][0]}"

    def test_integrator_negative_input(self):
        """Test integrator with negative input values."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        dtime = 0.1
        params = {'init_conds': 5.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        # Run 10 steps with constant negative input
        for i in range(10):
            result = block.execute(time=i*dtime, inputs={0: np.array([-1.0])}, params=params, dtime=dtime)

        # Should decrease from 5.0 by 10 * 0.1 * 1.0 = 1.0, so result is 4.0
        expected = 5.0 - 10 * dtime * 1.0
        assert np.isclose(result[0][0], expected, atol=0.1), f"Expected {expected}, got {result[0][0]}"

    def test_integrator_zero_input(self):
        """Test integrator with zero input stays at initial condition."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        dtime = 0.1
        params = {'init_conds': 3.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        # Run 5 steps with zero input
        for i in range(5):
            result = block.execute(time=i*dtime, inputs={0: np.array([0.0])}, params=params, dtime=dtime)

        # Should stay at initial condition
        assert np.isclose(result[0][0], 3.0, atol=0.01), f"Expected 3.0, got {result[0][0]}"

    def test_integrator_output_only_mode(self):
        """Test output_only mode returns current value without integrating."""
        from blocks.integrator import IntegratorBlock
        block = IntegratorBlock()
        dtime = 0.1
        params = {'init_conds': 2.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        # Initialize
        block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=dtime)
        val_after_init = params['mem'][0]

        # Call output_only - should not change mem
        result = block.execute(time=dtime, inputs={0: np.array([1.0])}, params=params, dtime=dtime, output_only=True)
        val_after_output_only = params['mem'][0]

        # mem should be unchanged after output_only call
        assert val_after_output_only == val_after_init, "output_only should not change internal state"
