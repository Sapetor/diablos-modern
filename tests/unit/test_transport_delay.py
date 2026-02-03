import pytest
import numpy as np


@pytest.mark.unit
class TestTransportDelay:
    """Tests for TransportDelay block."""

    def test_block_properties(self):
        """Test basic block properties and metadata."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        assert block.block_name == 'TransportDelay'
        assert block.category == 'Control'
        assert block.color == 'cyan'
        assert 'delay_time' in block.params
        assert 'initial_value' in block.params
        assert len(block.inputs) == 1
        assert len(block.outputs) == 1

    def test_zero_delay_passthrough(self):
        """With delay=0, output should equal input after buffer builds."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        params = {'delay_time': 0.0, 'initial_value': 0.0, '_init_start_': True}

        # First call - initialization
        result = block.execute(0.0, {0: np.array([1.0])}, params)
        # With zero delay but only one sample, should return initial value
        assert result[0][0] == 0.0

        # Second call - now we have two samples for interpolation
        result = block.execute(0.01, {0: np.array([2.0])}, params)
        # With zero delay and two samples, should interpolate to current
        assert result[0][0] == 2.0

        # Third call - continues to track input
        result = block.execute(0.02, {0: np.array([3.0])}, params)
        assert result[0][0] == 3.0

    def test_constant_delay(self):
        """Signal should arrive after delay time."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        delay = 0.1
        params = {'delay_time': delay, 'initial_value': 0.0, '_init_start_': True}

        # Step input at t=0: u(t) = 1.0
        dt = 0.01
        times = np.arange(0.0, 0.3, dt)
        outputs = []

        for t in times:
            result = block.execute(t, {0: np.array([1.0])}, params)
            outputs.append(result[0][0])

        outputs = np.array(outputs)

        # Before delay time, output should be initial value (0.0)
        before_delay_idx = int(delay / dt)
        assert np.all(outputs[:before_delay_idx] == 0.0)

        # After delay time, output should be 1.0 (with some tolerance for interpolation)
        after_delay_idx = int((delay + 0.05) / dt)  # 50ms after delay
        assert np.all(outputs[after_delay_idx:] > 0.99)

    def test_initial_value_before_delay(self):
        """Before delay elapses, should return initial_value."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        initial_val = 5.0
        delay = 0.2
        params = {'delay_time': delay, 'initial_value': initial_val, '_init_start_': True}

        # Execute at early times (before delay)
        result1 = block.execute(0.0, {0: np.array([10.0])}, params)
        assert result1[0][0] == initial_val

        result2 = block.execute(0.05, {0: np.array([10.0])}, params)
        assert result2[0][0] == initial_val

        result3 = block.execute(0.15, {0: np.array([10.0])}, params)
        assert result3[0][0] == initial_val

    def test_interpolation_accuracy(self):
        """Test linear interpolation for sub-timestep delays."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        delay = 0.05  # 50ms delay
        params = {'delay_time': delay, 'initial_value': 0.0, '_init_start_': True}

        # Create a ramp signal: u(t) = t
        dt = 0.01
        times = np.arange(0.0, 0.2, dt)

        for t in times:
            _ = block.execute(t, {0: np.array([t])}, params)

        # Now check interpolation at a specific time
        t_test = 0.15
        result = block.execute(t_test, {0: np.array([t_test])}, params)

        # Expected output: u(t - delay) = t_test - delay
        expected = t_test - delay
        actual = result[0][0]

        # Should be very close due to linear interpolation of linear signal
        assert abs(actual - expected) < 0.01

    def test_ramp_signal(self):
        """Test with linearly increasing input."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        delay = 0.1
        params = {'delay_time': delay, 'initial_value': 0.0, '_init_start_': True}

        # Ramp: u(t) = 10*t
        dt = 0.01
        times = np.arange(0.0, 0.5, dt)
        outputs = []

        for t in times:
            input_val = 10.0 * t
            result = block.execute(t, {0: np.array([input_val])}, params)
            outputs.append(result[0][0])

        outputs = np.array(outputs)

        # After sufficient time, output should be delayed input
        # At t=0.3, output should be approximately u(0.3-0.1) = u(0.2) = 2.0
        idx_030 = int(0.3 / dt)
        expected_030 = 10.0 * (0.3 - delay)
        assert abs(outputs[idx_030] - expected_030) < 0.2

    def test_buffer_pruning(self):
        """Verify buffer doesn't grow unbounded."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        delay = 0.1
        params = {'delay_time': delay, 'initial_value': 0.0, '_init_start_': True}

        # Run for a long time
        dt = 0.01
        for i in range(1000):
            t = i * dt
            _ = block.execute(t, {0: np.array([1.0])}, params)

        # Buffer should be pruned to reasonable size
        # Should keep ~2x delay worth of samples
        max_expected_length = int(2.0 * delay / dt) + 10  # Add margin
        assert len(params['_time_buffer_']) < max_expected_length
        assert len(params['_value_buffer_']) < max_expected_length
        assert len(params['_time_buffer_']) == len(params['_value_buffer_'])

    def test_step_response(self):
        """Test classic step response with delay."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        delay = 0.1
        params = {'delay_time': delay, 'initial_value': 0.0, '_init_start_': True}

        dt = 0.01
        step_time = 0.05

        outputs = []
        for i in range(30):
            t = i * dt
            # Step input: 0 before step_time, 1 after
            input_val = 1.0 if t >= step_time else 0.0
            result = block.execute(t, {0: np.array([input_val])}, params)
            outputs.append(result[0][0])

        outputs = np.array(outputs)

        # Output should step at step_time + delay
        step_output_time = step_time + delay
        idx_before = int((step_output_time - 0.02) / dt)
        idx_after = int((step_output_time + 0.02) / dt)

        assert outputs[idx_before] < 0.1
        assert outputs[idx_after] > 0.9

    def test_vector_input(self):
        """Test with vector-valued inputs."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        delay = 0.05
        params = {'delay_time': delay, 'initial_value': 0.0, '_init_start_': True}

        # Vector input - build up buffer first
        dt = 0.01
        for i in range(20):
            t = i * dt
            input_vec = np.array([t, 2*t, 3*t])
            result = block.execute(t, {0: input_vec}, params)

            target_time = t - delay

            # Before delay or when target_time <= first buffer entry: returns initial_value (1,)
            # When target_time is strictly between buffer entries: vector preserved (3,)
            if target_time <= 0.0:  # Includes boundary case where target_time == time_buffer[0]
                assert result[0].shape == (1,)
            elif i > 5:  # After enough buffer built up and target_time > time_buffer[0]
                assert result[0].shape == (3,)

        # Check final delayed output
        t_final = 0.15
        result = block.execute(t_final, {0: np.array([t_final, 2*t_final, 3*t_final])}, params)

        # Expected: delayed by 0.05s
        expected = np.array([t_final - delay, 2*(t_final - delay), 3*(t_final - delay)])
        np.testing.assert_allclose(result[0], expected, atol=0.02)

    def test_negative_delay_clamped_to_zero(self):
        """Negative delay should be clamped to zero."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        params = {'delay_time': -0.1, 'initial_value': 0.0, '_init_start_': True}

        # Should behave like zero delay
        result1 = block.execute(0.0, {0: np.array([1.0])}, params)
        result2 = block.execute(0.01, {0: np.array([2.0])}, params)

        # With clamped zero delay, should pass through after buffer builds
        assert result2[0][0] == 2.0

    def test_initialization_flag(self):
        """Test that _init_start_ flag properly resets buffers."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        params = {'delay_time': 0.1, 'initial_value': 0.0, '_init_start_': True}

        # First execution
        block.execute(0.0, {0: np.array([1.0])}, params)
        assert not params['_init_start_']
        assert len(params['_time_buffer_']) == 1

        # Reset initialization flag
        params['_init_start_'] = True
        block.execute(0.1, {0: np.array([2.0])}, params)

        # Buffers should be reset
        assert not params['_init_start_']
        assert len(params['_time_buffer_']) == 1

    def test_sine_wave_delay(self):
        """Test delay on sinusoidal signal."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        delay = 0.1
        freq = 5.0  # 5 Hz
        params = {'delay_time': delay, 'initial_value': 0.0, '_init_start_': True}

        dt = 0.001
        times = np.arange(0.0, 0.5, dt)
        outputs = []

        for t in times:
            input_val = np.sin(2 * np.pi * freq * t)
            result = block.execute(t, {0: np.array([input_val])}, params)
            outputs.append(result[0][0])

        outputs = np.array(outputs)

        # After delay has elapsed, check phase shift
        # At t=0.3, output should be sin(2*pi*freq*(0.3-delay))
        idx = int(0.3 / dt)
        expected = np.sin(2 * np.pi * freq * (0.3 - delay))
        assert abs(outputs[idx] - expected) < 0.05

    def test_no_input_uses_initial_value(self):
        """When no input is provided, should use initial_value."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        params = {'delay_time': 0.1, 'initial_value': 7.5, '_init_start_': True}

        # No input provided (empty dict)
        result = block.execute(0.0, {}, params)
        assert result[0][0] == 7.5

    def test_buffer_consistency(self):
        """Verify time and value buffers stay synchronized."""
        from blocks.transport_delay import TransportDelayBlock
        block = TransportDelayBlock()
        params = {'delay_time': 0.1, 'initial_value': 0.0, '_init_start_': True}

        dt = 0.01
        for i in range(50):
            t = i * dt
            _ = block.execute(t, {0: np.array([float(i)])}, params)

            # Buffers should always have same length
            assert len(params['_time_buffer_']) == len(params['_value_buffer_'])

            # Time buffer should be monotonically increasing
            if len(params['_time_buffer_']) > 1:
                assert all(params['_time_buffer_'][i] <= params['_time_buffer_'][i+1]
                          for i in range(len(params['_time_buffer_'])-1))
