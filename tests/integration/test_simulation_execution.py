"""
End-to-end integration tests for simulation execution.

These tests create blocks programmatically, connect them, run simulation steps,
and verify outputs to ensure the complete simulation pipeline works correctly.
"""

import pytest
import numpy as np


@pytest.mark.integration
class TestSimulationExecution:
    """Test complete simulation workflows."""

    def test_constant_to_scope(self):
        """Simple diagram: Constant → Scope, verify collected values."""
        from blocks.constant import ConstantBlock
        from blocks.scope import ScopeBlock

        const_block = ConstantBlock()
        scope_block = ScopeBlock()

        const_params = {'value': 5.0}
        scope_params = {'labels': 'const_out', '_init_start_': True, '_name_': 'TestScope'}

        # Run 10 simulation steps
        for i in range(10):
            time = i * 0.1
            const_out = const_block.execute(time=time, inputs={}, params=const_params)
            scope_block.execute(time=time, inputs={0: const_out[0]}, params=scope_params)

        # Verify scope collected correct values
        vector = scope_params['vector']
        assert len(vector) == 10, f"Should have 10 data points, got {len(vector)}"
        assert all(np.isclose(v, 5.0) for v in vector), "All values should be 5.0"

    def test_step_to_scope(self):
        """Test step signal collection."""
        from blocks.step import StepBlock
        from blocks.scope import ScopeBlock

        step_block = StepBlock()
        scope_block = ScopeBlock()

        step_params = {'value': 1.0, 'delay': 0.5, 'type': 'up',
                      'pulse_start_up': True, '_init_start_': True}
        scope_params = {'labels': 'step_out', '_init_start_': True, '_name_': 'TestScope'}

        times = np.linspace(0, 1.0, 11)
        for t in times:
            step_out = step_block.execute(time=t, inputs={}, params=step_params)
            scope_block.execute(time=t, inputs={0: step_out[0][0]}, params=scope_params)

        vector = scope_params['vector']
        # Before t=0.5: values should be 0
        # After t=0.5: values should be 1.0
        assert vector[0] == 0.0, "Step should be 0 before delay"
        assert vector[-1] == 1.0, "Step should be 1.0 after delay"

    def test_sine_amplitude(self):
        """Test sine wave amplitude is correct."""
        from blocks.sine import SineBlock

        sine_block = SineBlock()
        params = {'amplitude': 3.0, 'omega': 2*np.pi, 'init_angle': 0.0}

        # Collect values over one period
        values = []
        for t in np.linspace(0, 1.0, 100):
            result = sine_block.execute(time=t, inputs={}, params=params)
            values.append(result[0])

        assert np.isclose(max(values), 3.0, atol=0.1), "Max should be amplitude"
        assert np.isclose(min(values), -3.0, atol=0.1), "Min should be -amplitude"

    def test_gain_block(self):
        """Test gain block multiplies correctly."""
        from blocks.gain import GainBlock

        gain_block = GainBlock()
        params = {'gain': 2.5}

        result = gain_block.execute(time=0.0, inputs={0: 4.0}, params=params)
        assert np.isclose(result[0], 10.0), "Gain should multiply input by k"

    def test_sum_block(self):
        """Test sum block adds/subtracts correctly."""
        from blocks.sum import SumBlock

        sum_block = SumBlock()
        params = {'sign': '+-', '_init_start_': True}

        result = sum_block.execute(time=0.0, inputs={0: 5.0, 1: 3.0}, params=params)
        assert np.isclose(result[0], 2.0), "Sum with +- should give 5-3=2"

    def test_sum_three_inputs(self):
        """Test sum block with three inputs."""
        from blocks.sum import SumBlock

        sum_block = SumBlock()
        params = {'sign': '++-'}

        result = sum_block.execute(time=0.0, inputs={0: 10.0, 1: 5.0, 2: 3.0}, params=params)
        assert np.isclose(result[0], 12.0), "Sum with ++- should give 10+5-3=12"

    def test_integrator_ramp(self):
        """Test integrator produces ramp from constant input."""
        from blocks.integrator import IntegratorBlock

        integrator = IntegratorBlock()
        dtime = 0.01
        params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        # Integrate constant 1.0 for 1 second
        for i in range(100):
            result = integrator.execute(
                time=i*dtime,
                inputs={0: np.array([1.0])},
                params=params,
                dtime=dtime
            )

        # After 100 steps of dt=0.01 with input 1.0: integral ≈ 1.0
        assert np.isclose(result[0][0], 1.0, atol=0.05), f"Integral should be ~1.0, got {result[0][0]}"

    def test_integrator_initial_condition(self):
        """Test integrator respects initial conditions."""
        from blocks.integrator import IntegratorBlock

        integrator = IntegratorBlock()
        params = {'init_conds': 5.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        # First step with zero input should return initial condition
        result = integrator.execute(time=0.0, inputs={0: np.array([0.0])}, params=params, dtime=0.01)
        assert np.isclose(result[0][0], 5.0, atol=0.01), f"Initial output should be 5.0, got {result[0][0]}"

    def test_chain_constant_gain_scope(self):
        """Test chain: Constant → Gain → Scope."""
        from blocks.constant import ConstantBlock
        from blocks.gain import GainBlock
        from blocks.scope import ScopeBlock

        const_block = ConstantBlock()
        gain_block = GainBlock()
        scope_block = ScopeBlock()

        const_params = {'value': 3.0}
        gain_params = {'gain': 4.0}
        scope_params = {'labels': 'out', '_init_start_': True, '_name_': 'TestScope'}

        # Run 5 steps
        for i in range(5):
            const_out = const_block.execute(time=i*0.1, inputs={}, params=const_params)
            gain_out = gain_block.execute(time=i*0.1, inputs={0: const_out[0]}, params=gain_params)
            scope_block.execute(time=i*0.1, inputs={0: gain_out[0]}, params=scope_params)

        vector = scope_params['vector']
        assert all(np.isclose(v, 12.0) for v in vector), "All values should be 3.0 * 4.0 = 12.0"

    def test_chain_sine_gain_sum_scope(self):
        """Test more complex chain: Sine → Gain → Sum (with Constant) → Scope."""
        from blocks.sine import SineBlock
        from blocks.constant import ConstantBlock
        from blocks.gain import GainBlock
        from blocks.sum import SumBlock
        from blocks.scope import ScopeBlock

        sine_block = SineBlock()
        const_block = ConstantBlock()
        gain_block = GainBlock()
        sum_block = SumBlock()
        scope_block = ScopeBlock()

        sine_params = {'amplitude': 1.0, 'omega': 2*np.pi, 'init_angle': 0.0}
        const_params = {'value': 2.0}
        gain_params = {'gain': 3.0}
        sum_params = {'sign': '++'}
        scope_params = {'labels': 'out', '_init_start_': True, '_name_': 'TestScope'}

        # Run over one period
        times = np.linspace(0, 1.0, 20)
        for t in times:
            # Sine → Gain (scaled sine)
            sine_out = sine_block.execute(time=t, inputs={}, params=sine_params)
            gain_out = gain_block.execute(time=t, inputs={0: sine_out[0]}, params=gain_params)

            # Constant
            const_out = const_block.execute(time=t, inputs={}, params=const_params)

            # Sum (gain_out + const_out)
            sum_out = sum_block.execute(time=t, inputs={0: gain_out[0], 1: const_out[0]}, params=sum_params)

            # Scope
            scope_block.execute(time=t, inputs={0: sum_out[0]}, params=scope_params)

        vector = scope_params['vector']
        # At t=0: sin(0) = 0, so output = 3*0 + 2 = 2
        assert np.isclose(vector[0], 2.0, atol=0.01), f"At t=0, output should be 2.0, got {vector[0]}"

        # Check that output varies (sine wave effect)
        assert np.std(vector) > 0.5, "Output should vary due to sine input"

    def test_feedback_loop_integrator(self):
        """Test simple feedback loop with integrator."""
        from blocks.constant import ConstantBlock
        from blocks.sum import SumBlock
        from blocks.integrator import IntegratorBlock
        from blocks.gain import GainBlock

        # System: input → sum → integrator → output (with feedback through gain)
        # This simulates: dx/dt = u - k*x, where u is constant input

        const_block = ConstantBlock()
        sum_block = SumBlock()
        integrator = IntegratorBlock()
        gain_block = GainBlock()

        const_params = {'value': 1.0}
        sum_params = {'sign': '+-'}
        int_params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}
        gain_params = {'gain': 0.5}  # Feedback gain

        dt = 0.01
        x = 0.0  # Track integrator output

        # Simulate for 2 seconds
        for i in range(200):
            t = i * dt

            # Constant input
            const_out = const_block.execute(time=t, inputs={}, params=const_params)

            # Feedback
            feedback = gain_block.execute(time=t, inputs={0: x}, params=gain_params)

            # Sum (input - feedback)
            sum_out = sum_block.execute(time=t, inputs={0: const_out[0], 1: feedback[0]}, params=sum_params)

            # Integrator
            int_out = integrator.execute(time=t, inputs={0: sum_out[0]}, params=int_params, dtime=dt)
            x = int_out[0][0]

        # Should approach steady-state (exponential approach to u/k)
        # With forward Euler and these parameters, it converges more slowly
        assert x > 1.0, f"Should be increasing from initial condition, got {x}"
        assert x < 2.5, f"Should not overshoot significantly, got {x}"

    def test_parallel_paths(self):
        """Test signal splitting and parallel processing."""
        from blocks.constant import ConstantBlock
        from blocks.gain import GainBlock
        from blocks.sum import SumBlock

        # Topology: Constant → (Gain1, Gain2) → Sum
        # Output should be constant * (gain1 + gain2)

        const_block = ConstantBlock()
        gain1 = GainBlock()
        gain2 = GainBlock()
        sum_block = SumBlock()

        const_params = {'value': 5.0}
        gain1_params = {'gain': 2.0}
        gain2_params = {'gain': 3.0}
        sum_params = {'sign': '++'}

        # Single time step
        const_out = const_block.execute(time=0.0, inputs={}, params=const_params)

        # Parallel paths
        gain1_out = gain1.execute(time=0.0, inputs={0: const_out[0]}, params=gain1_params)
        gain2_out = gain2.execute(time=0.0, inputs={0: const_out[0]}, params=gain2_params)

        # Sum
        sum_out = sum_block.execute(time=0.0, inputs={0: gain1_out[0], 1: gain2_out[0]}, params=sum_params)

        expected = 5.0 * (2.0 + 3.0)  # 25.0
        assert np.isclose(sum_out[0], expected), f"Expected {expected}, got {sum_out[0]}"

    def test_step_response_first_order_system(self):
        """Test step response of first-order system: G(s) = K/(τs + 1)."""
        from blocks.step import StepBlock
        from blocks.sum import SumBlock
        from blocks.gain import GainBlock
        from blocks.integrator import IntegratorBlock

        # Build system: Step → Sum → Integrator → (output and feedback via gain)
        # Represents: τ*dy/dt + y = K*u, or dy/dt = (K*u - y)/τ

        step_block = StepBlock()
        gain_k = GainBlock()  # Process gain K
        sum_block = SumBlock()
        gain_tau_inv = GainBlock()  # 1/τ
        integrator = IntegratorBlock()

        K = 2.0
        tau = 0.5

        step_params = {'value': 1.0, 'delay': 0.1, 'type': 'up',
                      'pulse_start_up': True, '_init_start_': True}
        gain_k_params = {'gain': K}
        sum_params = {'sign': '+-'}
        gain_tau_params = {'gain': 1.0/tau}
        int_params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        dt = 0.005
        y = 0.0
        outputs = []
        times = []

        # Simulate for 2 seconds
        for i in range(400):
            t = i * dt
            times.append(t)

            # Step input
            step_out = step_block.execute(time=t, inputs={}, params=step_params)

            # K*u
            scaled_input = gain_k.execute(time=t, inputs={0: step_out[0]}, params=gain_k_params)

            # K*u - y
            error = sum_block.execute(time=t, inputs={0: scaled_input[0], 1: y}, params=sum_params)

            # (K*u - y)/τ
            deriv = gain_tau_inv.execute(time=t, inputs={0: error[0]}, params=gain_tau_params)

            # Integrate
            int_out = integrator.execute(time=t, inputs={0: deriv[0]}, params=int_params, dtime=dt)
            y = int_out[0][0]

            outputs.append(y)

        # Check steady-state (should approach K*step_value = 2.0)
        steady_state = np.mean(outputs[-50:])
        assert np.isclose(steady_state, K, atol=0.1), f"Steady-state should be ~{K}, got {steady_state}"

        # Check response started from 0
        assert outputs[0] < 0.1, "Should start near zero"

    def test_multiple_scopes(self):
        """Test multiple scope blocks collecting data simultaneously."""
        from blocks.sine import SineBlock
        from blocks.gain import GainBlock
        from blocks.scope import ScopeBlock

        sine_block = SineBlock()
        gain_block = GainBlock()
        scope1 = ScopeBlock()
        scope2 = ScopeBlock()

        sine_params = {'amplitude': 1.0, 'omega': 2*np.pi, 'init_angle': 0.0}
        gain_params = {'gain': 2.0}
        scope1_params = {'labels': 'sine', '_init_start_': True, '_name_': 'Scope1'}
        scope2_params = {'labels': 'scaled', '_init_start_': True, '_name_': 'Scope2'}

        times = np.linspace(0, 1.0, 50)
        for t in times:
            sine_out = sine_block.execute(time=t, inputs={}, params=sine_params)
            gain_out = gain_block.execute(time=t, inputs={0: sine_out[0]}, params=gain_params)

            scope1.execute(time=t, inputs={0: sine_out[0]}, params=scope1_params)
            scope2.execute(time=t, inputs={0: gain_out[0]}, params=scope2_params)

        vec1 = scope1_params['vector']
        vec2 = scope2_params['vector']

        # Both should have same length
        assert len(vec1) == len(vec2) == 50, "Both scopes should collect 50 points"

        # Scope2 should be 2x Scope1
        for v1, v2 in zip(vec1, vec2):
            assert np.isclose(v2, 2.0 * v1, atol=0.01), "Scope2 should be 2x Scope1"

    def test_vector_signals(self):
        """Test blocks handling vector signals."""
        from blocks.constant import ConstantBlock
        from blocks.gain import GainBlock

        const_block = ConstantBlock()
        gain_block = GainBlock()

        # Vector constant
        const_params = {'value': [1.0, 2.0, 3.0]}
        gain_params = {'gain': 2.0}

        const_out = const_block.execute(time=0.0, inputs={}, params=const_params)
        gain_out = gain_block.execute(time=0.0, inputs={0: const_out[0]}, params=gain_params)

        expected = np.array([2.0, 4.0, 6.0])
        assert np.allclose(gain_out[0], expected), f"Expected {expected}, got {gain_out[0]}"

    def test_integrator_vector_input(self):
        """Test integrator with vector inputs."""
        from blocks.integrator import IntegratorBlock

        integrator = IntegratorBlock()
        params = {'init_conds': [0.0, 0.0], 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'VecInt'}

        dt = 0.1
        input_vec = np.array([1.0, 2.0])

        # Integrate for 10 steps
        for i in range(10):
            result = integrator.execute(time=i*dt, inputs={0: input_vec}, params=params, dtime=dt)

        # After 10 steps with dt=0.1: integral ≈ [1.0, 2.0]
        expected = np.array([1.0, 2.0])
        assert np.allclose(result[0], expected, atol=0.1), f"Expected {expected}, got {result[0]}"

    def test_time_varying_input(self):
        """Test system with time-varying input (ramp)."""
        from blocks.ramp import RampBlock
        from blocks.integrator import IntegratorBlock

        ramp_block = RampBlock()
        integrator = IntegratorBlock()

        ramp_params = {'slope': 2.0, 'delay': 0.0}
        int_params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}

        dt = 0.01

        # Integrate ramp for 1 second
        # ramp(t) = 2t, so integral should be t^2
        for i in range(100):
            t = i * dt
            ramp_out = ramp_block.execute(time=t, inputs={}, params=ramp_params)
            int_out = integrator.execute(time=t, inputs={0: ramp_out[0]}, params=int_params, dtime=dt)

        # At t=1.0, integral of 2t should be t^2 = 1.0
        expected = 1.0
        assert np.isclose(int_out[0][0], expected, atol=0.1), f"Expected {expected}, got {int_out[0][0]}"

    def test_zero_crossing(self):
        """Test system crossing zero (sine wave)."""
        from blocks.sine import SineBlock

        sine_block = SineBlock()
        params = {'amplitude': 1.0, 'omega': 2*np.pi, 'init_angle': 0.0}

        # Collect values around half-period
        times = np.linspace(0, 0.6, 20)
        values = []

        for t in times:
            result = sine_block.execute(time=t, inputs={}, params=params)
            values.append(result[0])

        # Should cross zero somewhere in the middle
        # Check that sign changes
        signs = [np.sign(v) for v in values]
        assert 1 in signs and -1 in signs, "Sine should cross zero (have both positive and negative values)"

    def test_saturation_behavior(self):
        """Test gain doesn't saturate (linear behavior maintained)."""
        from blocks.gain import GainBlock

        gain_block = GainBlock()
        params = {'gain': 10.0}

        # Test with various input magnitudes
        inputs_test = [-100, -10, -1, 0, 1, 10, 100]

        for inp in inputs_test:
            result = gain_block.execute(time=0.0, inputs={0: float(inp)}, params=params)
            expected = inp * 10.0
            assert np.isclose(result[0], expected), f"Gain should be linear: {inp} * 10 = {expected}, got {result[0]}"


@pytest.mark.integration
class TestSimulationEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_time_step(self):
        """Test blocks at t=0."""
        from blocks.sine import SineBlock
        from blocks.step import StepBlock

        sine_block = SineBlock()
        step_block = StepBlock()

        sine_params = {'amplitude': 1.0, 'omega': 1.0, 'init_angle': 0.0}
        step_params = {'value': 1.0, 'delay': 0.0, 'type': 'up',
                      'pulse_start_up': True, '_init_start_': True}

        sine_result = sine_block.execute(time=0.0, inputs={}, params=sine_params)
        step_result = step_block.execute(time=0.0, inputs={}, params=step_params)

        assert np.isclose(sine_result[0], 0.0), "Sine at t=0 should be 0"
        # Step 'up' type with delay=0 is already high at t=0 (change=False when t>=delay)
        assert step_result[0][0] == 1.0, "Step 'up' at t=0 with delay=0 should be at final value"

    def test_large_time_values(self):
        """Test blocks with large time values."""
        from blocks.sine import SineBlock

        sine_block = SineBlock()
        params = {'amplitude': 1.0, 'omega': 1.0, 'init_angle': 0.0}

        # Should handle large time values without numerical issues
        for t in [100.0, 1000.0, 10000.0]:
            result = sine_block.execute(time=t, inputs={}, params=params)
            assert -1.0 <= result[0] <= 1.0, f"Sine should stay bounded at t={t}"

    def test_negative_gain(self):
        """Test gain with negative value (signal inversion)."""
        from blocks.gain import GainBlock

        gain_block = GainBlock()
        params = {'gain': -2.0}

        result = gain_block.execute(time=0.0, inputs={0: 5.0}, params=params)
        assert np.isclose(result[0], -10.0), "Negative gain should invert and scale"

    def test_zero_gain(self):
        """Test gain block with zero gain."""
        from blocks.gain import GainBlock

        gain_block = GainBlock()
        params = {'gain': 0.0}

        result = gain_block.execute(time=0.0, inputs={0: 100.0}, params=params)
        assert np.isclose(result[0], 0.0), "Zero gain should output zero"

    def test_rapid_oscillation(self):
        """Test sine with very high frequency."""
        from blocks.sine import SineBlock

        sine_block = SineBlock()
        params = {'amplitude': 1.0, 'omega': 1000.0, 'init_angle': 0.0}

        # Should still be bounded
        for t in np.linspace(0, 0.1, 10):
            result = sine_block.execute(time=t, inputs={}, params=params)
            assert -1.0 <= result[0] <= 1.0, f"High-frequency sine should stay bounded at t={t}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
