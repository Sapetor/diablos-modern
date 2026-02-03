"""Unit tests for DiscreteTransferFunction block."""

import pytest
import numpy as np


@pytest.mark.unit
class TestDiscreteTransferFunction:
    """Tests for DiscreteTranFn block."""

    def test_block_properties(self):
        """Test basic block properties."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        assert block.block_name == 'DiscreteTranFn'
        assert block.fn_name == 'discrete_transfer_function'
        assert block.b_type == 2  # Proper system with direct feedthrough

    def test_default_params(self):
        """Test default parameter values."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = block.params

        assert params['numerator']['default'] == [1.0, 0.0]
        assert params['denominator']['default'] == [1.0, -0.5]
        assert params['sampling_time']['default'] == -1.0
        assert params['_init_start_']['default'] is True

    def test_unity_gain(self):
        """H(z) = 1/1 should pass input through."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [1.0],
            'denominator': [1.0],
            '_init_start_': True
        }

        # First call initializes
        result = block.execute(0.0, {0: 5.0}, params)
        assert result['E'] is False
        assert result[0] == pytest.approx(5.0)

        # Second call with different input
        result = block.execute(0.1, {0: 3.0}, params)
        assert result['E'] is False
        assert result[0] == pytest.approx(3.0)

    def test_first_order_lowpass(self):
        """Test simple first-order filter response.

        H(z) = 0.5 / (1 - 0.5*z^-1)

        State-space: x[k+1] = 0.5*x[k] + u[k]
                     y[k] = 0.5*x[k]

        With zero initial state and unit step input:
        k=0: x=0, y=0, x_next=1
        k=1: x=1, y=0.5, x_next=1.5
        k=2: x=1.5, y=0.75, x_next=1.75
        k=3: x=1.75, y=0.875, ...
        """
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [0.5],
            'denominator': [1.0, -0.5],
            '_init_start_': True
        }

        # Step input: u[k] = 1.0 for all k
        outputs = []
        for k in range(10):
            result = block.execute(float(k), {0: 1.0}, params)
            assert result['E'] is False
            outputs.append(result[0])

        # Expected output sequence (delayed by 1 sample due to tf2ss conversion)
        # y[k] = (1 - 0.5^(k+1)) for k >= 1, y[0] = 0
        expected = [0.0] + [1 - 0.5**(k+1) for k in range(9)]

        for i, (got, exp) in enumerate(zip(outputs, expected)):
            assert got == pytest.approx(exp, abs=1e-10), f"Step {i}: got {got}, expected {exp}"

    def test_integrator_discrete(self):
        """Test discrete integrator: H(z) = Ts*z^-1 / (1 - z^-1)

        This is a backward Euler integrator with sampling time Ts.
        y[k] = y[k-1] + Ts*u[k-1]
        """
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        Ts = 0.1
        params = {
            'numerator': [0.0, Ts],  # Ts*z^-1
            'denominator': [1.0, -1.0],  # 1 - z^-1
            '_init_start_': True
        }

        # Constant input u[k] = 2.0
        outputs = []
        for k in range(5):
            result = block.execute(float(k)*Ts, {0: 2.0}, params)
            assert result['E'] is False
            outputs.append(result[0])

        # Expected: y[0] = 0, y[1] = 0.2, y[2] = 0.4, y[3] = 0.6, y[4] = 0.8
        expected = [0.0, 0.2, 0.4, 0.6, 0.8]

        for i, (got, exp) in enumerate(zip(outputs, expected)):
            assert got == pytest.approx(exp, abs=1e-10), f"Step {i}: got {got}, expected {exp}"

    def test_state_update(self):
        """Verify state evolves correctly over multiple steps.

        H(z) = (z + 0.5) / (z - 0.8)
        In z^-1 form: H(z) = (1 + 0.5*z^-1) / (1 - 0.8*z^-1)
        """
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [1.0, 0.5],
            'denominator': [1.0, -0.8],
            '_init_start_': True
        }

        # Impulse response: u[0] = 1, u[k] = 0 for k > 0
        outputs = []
        for k in range(6):
            u = 1.0 if k == 0 else 0.0
            result = block.execute(float(k), {0: u}, params)
            assert result['E'] is False
            outputs.append(result[0])

        # Impulse response should decay with pole at 0.8
        # y[0] = 1.0, y[1] = 0.8 + 0.5 = 1.3, y[2] = 0.8*1.3 = 1.04, ...
        # Analytical: y[k] = 1.0*0.8^k + 0.5*0.8^(k-1) for k >= 1
        assert outputs[0] == pytest.approx(1.0, abs=1e-10)
        assert outputs[1] == pytest.approx(1.3, abs=1e-10)
        assert outputs[2] == pytest.approx(1.04, abs=1e-10)

        # Check state vector is being updated
        assert '_x_' in params
        assert params['_x_'].shape[0] == 1  # First order system

    def test_held_output_between_samples(self):
        """With sampling_time > 0, output should be held between sample times."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        Ts = 0.1
        params = {
            'numerator': [1.0],
            'denominator': [1.0, -0.5],
            'sampling_time': Ts,
            '_init_start_': True
        }

        # At t=0.0, sample and update
        result = block.execute(0.0, {0: 1.0}, params)
        assert result['E'] is False
        output_t0 = result[0]

        # At t=0.05 (between samples), should hold previous output
        result = block.execute(0.05, {0: 5.0}, params)  # Input changed but shouldn't affect output
        assert result['E'] is False
        assert result[0] == pytest.approx(output_t0)
        assert params['_held_output_'] == pytest.approx(output_t0)

        # At t=0.1, should sample again
        result = block.execute(0.1, {0: 2.0}, params)
        assert result['E'] is False
        output_t1 = result[0]
        assert output_t1 != output_t0  # Should have updated

        # At t=0.15, should hold again
        result = block.execute(0.15, {0: 10.0}, params)
        assert result['E'] is False
        assert result[0] == pytest.approx(output_t1)

    def test_sampling_time_negative(self):
        """Negative sampling_time means continuous update (every call)."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [1.0],
            'denominator': [1.0, -0.5],
            'sampling_time': -1.0,  # Continuous
            '_init_start_': True
        }

        # Every call should update
        result1 = block.execute(0.0, {0: 1.0}, params)
        result2 = block.execute(0.01, {0: 2.0}, params)
        result3 = block.execute(0.02, {0: 3.0}, params)

        assert result1[0] != result2[0]
        assert result2[0] != result3[0]

    def test_error_handling_invalid_coeffs(self):
        """Test error return for invalid transfer function."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()

        # Empty denominator should cause error
        params = {
            'numerator': [1.0],
            'denominator': [],
            '_init_start_': True
        }

        result = block.execute(0.0, {0: 1.0}, params)
        assert result['E'] is True
        assert 'error' in result

    def test_zero_input(self):
        """Test with zero input - should decay to zero."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [1.0],
            'denominator': [1.0, -0.5],
            '_init_start_': True
        }

        # All zeros
        for k in range(5):
            result = block.execute(float(k), {0: 0.0}, params)
            assert result['E'] is False
            assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_output_only_mode(self):
        """Test output_only mode - should not update state."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [1.0],
            'denominator': [1.0, -0.5],
            '_init_start_': True
        }

        # Initialize
        result = block.execute(0.0, {0: 1.0}, params)
        state_after_init = params['_x_'].copy()

        # Call with output_only=True
        result = block.execute(0.1, {0: 5.0}, params, output_only=True)
        assert result['E'] is False

        # State should not have changed
        np.testing.assert_array_equal(params['_x_'], state_after_init)

    def test_higher_order_system(self):
        """Test second-order system.

        H(z) = (z^2 + 0.5*z) / (z^2 - 1.5*z + 0.5)
        In z^-1 form: H(z) = (1 + 0.5*z^-1) / (1 - 1.5*z^-1 + 0.5*z^-2)
        """
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [1.0, 0.5, 0.0],
            'denominator': [1.0, -1.5, 0.5],
            '_init_start_': True
        }

        # Execute a few steps
        outputs = []
        for k in range(5):
            u = 1.0 if k == 0 else 0.0
            result = block.execute(float(k), {0: u}, params)
            assert result['E'] is False
            outputs.append(result[0])

        # Should have 2 states
        assert params['_n_states_'] == 2
        assert params['_x_'].shape[0] == 2

        # First output should be nonzero (has direct feedthrough)
        assert outputs[0] != 0.0

    def test_initialization_with_init_conds(self):
        """Test initialization with custom initial conditions."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [1.0],
            'denominator': [1.0, -0.5],
            'init_conds': [2.0],  # Non-zero initial state
            '_init_start_': True
        }

        # First call with zero input - should see effect of initial condition
        result = block.execute(0.0, {0: 0.0}, params)
        assert result['E'] is False

        # Output should reflect initial state: y = C*x + D*u = C*2.0 + 0 = 2.0
        assert result[0] == pytest.approx(2.0, abs=1e-10)

    def test_multiple_initializations(self):
        """Test that re-initialization works correctly."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [1.0],
            'denominator': [1.0, -0.5],
            '_init_start_': True
        }

        # First run
        result1 = block.execute(0.0, {0: 1.0}, params)
        result2 = block.execute(0.1, {0: 1.0}, params)

        # Re-initialize
        params['_init_start_'] = True
        result3 = block.execute(0.0, {0: 1.0}, params)

        # Should get same result as first initialization
        assert result3[0] == pytest.approx(result1[0])

    def test_direct_feedthrough(self):
        """Test system with direct feedthrough (D != 0).

        H(z) = 2 - this is just H(z) = 2/1 with D=2
        """
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [2.0],
            'denominator': [1.0],
            '_init_start_': True
        }

        # Input should immediately affect output through D matrix
        result = block.execute(0.0, {0: 3.0}, params)
        assert result['E'] is False
        assert result[0] == pytest.approx(6.0)  # 2 * 3.0

    def test_no_input_provided(self):
        """Test behavior when no input is provided (defaults to 0)."""
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        block = DiscreteTransferFunctionBlock()
        params = {
            'numerator': [1.0],
            'denominator': [1.0, -0.5],
            '_init_start_': True
        }

        # No input in dictionary
        result = block.execute(0.0, {}, params)
        assert result['E'] is False
        assert result[0] == pytest.approx(0.0)
