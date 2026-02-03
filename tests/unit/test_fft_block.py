import pytest
import numpy as np


@pytest.mark.unit
class TestFFTBlock:
    """Tests for FFT block."""

    def test_block_properties(self):
        """Test basic block properties."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        assert block.block_name == 'FFT'
        assert block.category == 'Sinks'
        assert block.color == 'brown'

    def test_params_structure(self):
        """Test parameter definitions."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        assert 'window' in block.params
        assert 'title' in block.params
        assert 'normalize' in block.params
        assert 'log_scale' in block.params

    def test_window_parameter_options(self):
        """Test window parameter default and type."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        assert block.params['window']['default'] == 'hann'
        assert block.params['window']['type'] == 'string'

    def test_normalize_parameter(self):
        """Test normalize parameter."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        assert block.params['normalize']['default'] is True
        assert block.params['normalize']['type'] == 'bool'

    def test_log_scale_parameter(self):
        """Test log_scale parameter."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        assert block.params['log_scale']['default'] is False
        assert block.params['log_scale']['type'] == 'bool'

    def test_has_one_input(self):
        """FFT block should have one input port."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        assert len(block.inputs) == 1
        assert block.inputs[0]['name'] == 'in'

    def test_has_no_outputs(self):
        """FFT is a sink with no output ports."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        assert block.outputs == []

    def test_execute_returns_empty_dict(self):
        """FFT is a sink - returns empty dict."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}
        result = block.execute(0.0, {0: np.array([1.0])}, params)
        assert result == {}

    def test_buffer_initialization(self):
        """Test that first execute initializes buffers."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}

        assert '_fft_buffer_' not in params
        assert '_fft_time_' not in params

        block.execute(0.0, {0: np.array([1.0])}, params)

        assert '_fft_buffer_' in params
        assert '_fft_time_' in params
        assert isinstance(params['_fft_buffer_'], list)
        assert isinstance(params['_fft_time_'], list)

    def test_buffer_accumulation(self):
        """Test that execute accumulates samples in buffer."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}

        for i in range(10):
            block.execute(i * 0.1, {0: np.array([np.sin(i)])}, params)

        assert '_fft_buffer_' in params
        assert len(params['_fft_buffer_']) == 10
        assert len(params['_fft_time_']) == 10

    def test_time_buffer_accumulation(self):
        """Time values should be recorded correctly."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}

        times = [0.0, 0.1, 0.2, 0.3]
        for t in times:
            block.execute(t, {0: np.array([1.0])}, params)

        assert params['_fft_time_'] == times

    def test_scalar_input_storage(self):
        """Test storage of scalar input values."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}

        values = [1.0, 2.0, 3.0, 4.0]
        for i, val in enumerate(values):
            block.execute(i * 0.1, {0: np.array([val])}, params)

        assert len(params['_fft_buffer_']) == 4
        for i, val in enumerate(values):
            assert params['_fft_buffer_'][i] == val

    def test_vector_input_storage(self):
        """Test storage of vector input (takes first element)."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}

        # Vector inputs - should store the array
        block.execute(0.0, {0: np.array([1.0, 2.0, 3.0])}, params)
        block.execute(0.1, {0: np.array([4.0, 5.0, 6.0])}, params)

        assert len(params['_fft_buffer_']) == 2
        # For vector input, stores the whole array
        assert isinstance(params['_fft_buffer_'][0], np.ndarray)

    def test_empty_input_handling(self):
        """Test handling of empty/zero input."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}

        block.execute(0.0, {}, params)

        assert '_fft_buffer_' in params
        assert len(params['_fft_buffer_']) == 1
        assert params['_fft_buffer_'][0] == 0.0

    def test_continuous_accumulation(self):
        """Test that buffer continues to accumulate across multiple calls."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}

        # First batch
        for i in range(5):
            block.execute(i * 0.1, {0: np.array([i])}, params)

        assert len(params['_fft_buffer_']) == 5

        # Second batch - should continue accumulating
        for i in range(5, 10):
            block.execute(i * 0.1, {0: np.array([i])}, params)

        assert len(params['_fft_buffer_']) == 10

    def test_sine_wave_accumulation(self):
        """Test accumulation of a sine wave signal."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}

        # Generate sine wave samples
        fs = 100  # Sample rate
        freq = 5  # Hz
        duration = 1.0  # seconds
        t = np.arange(0, duration, 1/fs)

        for time_val in t:
            signal_val = np.sin(2 * np.pi * freq * time_val)
            block.execute(time_val, {0: np.array([signal_val])}, params)

        assert len(params['_fft_buffer_']) == len(t)
        assert len(params['_fft_time_']) == len(t)

        # Verify time values are stored correctly
        np.testing.assert_array_almost_equal(params['_fft_time_'], t)

    def test_doc_property(self):
        """Test documentation string."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        assert 'FFT' in block.doc or 'Spectrum' in block.doc
        assert 'frequency' in block.doc.lower() or 'Frequency' in block.doc

    def test_draw_icon_returns_path(self):
        """Test that draw_icon returns a QPainterPath."""
        from blocks.fft import FFTBlock
        from PyQt5.QtCore import QRectF

        block = FFTBlock()
        rect = QRectF(0, 0, 100, 100)
        path = block.draw_icon(rect)

        # Should return a QPainterPath
        from PyQt5.QtGui import QPainterPath
        assert isinstance(path, QPainterPath)

    def test_multiple_executions_same_params_dict(self):
        """Test that same params dict is used across executions."""
        from blocks.fft import FFTBlock
        block = FFTBlock()
        params = {}

        block.execute(0.0, {0: np.array([1.0])}, params)
        first_buffer_id = id(params['_fft_buffer_'])

        block.execute(0.1, {0: np.array([2.0])}, params)
        second_buffer_id = id(params['_fft_buffer_'])

        # Should be the same list object
        assert first_buffer_id == second_buffer_id
        assert len(params['_fft_buffer_']) == 2
