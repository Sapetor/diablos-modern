import numpy as np
from blocks.base_block import BaseBlock
import matplotlib.pyplot as plt


class FFTBlock(BaseBlock):
    """
    FFT Spectrum Analyzer block.
    Collects signal data during simulation and displays the frequency spectrum at the end.
    """

    @property
    def block_name(self):
        return "FFT"

    @property
    def category(self):
        return "Sinks"

    @property
    def color(self):
        return "brown"

    @property
    def doc(self):
        return (
            "Spectrum Analyzer (FFT)."
            "\n\nComputes and plots the Frequency Spectrum (Magnitude) of the input."
            "\n\nParameters:"
            "\n- Window: Tapering window (Hamming, Hanning, Rectangular)."
            "\n- Log Scale: Use logarithmic X/Y axes."
            "\n\nUsage:"
            "\nAnalyze frequency content, resonances, or noise."
        )

    @property
    def params(self):
        return {
            "title": {"type": "string", "default": "FFT Spectrum", "doc": "Plot title."},
            "window": {"type": "string", "default": "hann", "doc": "Window function: 'hann', 'hamming', 'blackman', 'none'."},
            "normalize": {"type": "bool", "default": True, "doc": "Normalize magnitude to 0-1."},
            "log_scale": {"type": "bool", "default": False, "doc": "Use dB scale for magnitude."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """Draw FFT spectrum icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Axis
        path.moveTo(0.15, 0.80)
        path.lineTo(0.15, 0.20)
        path.moveTo(0.15, 0.80)
        path.lineTo(0.85, 0.80)
        # Spectrum bars
        bars = [(0.22, 0.50), (0.32, 0.30), (0.42, 0.40), (0.52, 0.55), (0.62, 0.65), (0.72, 0.70)]
        for x, h in bars:
            path.moveTo(x, 0.80)
            path.lineTo(x, h)
            path.lineTo(x + 0.06, h)
            path.lineTo(x + 0.06, 0.80)
        return path

    def execute(self, time, inputs, params):
        # Get input signal value
        u = np.atleast_1d(inputs.get(0, 0)).astype(float)
        
        # Initialize buffer on first call
        if '_fft_buffer_' not in params:
            params['_fft_buffer_'] = []
            params['_fft_time_'] = []
        
        # Store signal value
        params['_fft_buffer_'].append(u[0] if len(u) == 1 else u)
        params['_fft_time_'].append(time)
        
        return {}

    def plot_spectrum(self, params):
        """Called after simulation to display the FFT spectrum."""
        buffer = params.get('_fft_buffer_', [])
        time_data = params.get('_fft_time_', [])
        
        if len(buffer) < 2:
            return
        
        # Convert to numpy array
        signal = np.array(buffer)
        if signal.ndim > 1:
            signal = signal[:, 0]  # Take first channel for multi-dimensional signals
        
        # Calculate sample rate
        if len(time_data) > 1:
            dt = np.mean(np.diff(time_data))
            fs = 1.0 / dt if dt > 0 else 1.0
        else:
            fs = 1.0
        
        # Apply window function
        window_type = params.get('window', 'hann')
        n = len(signal)
        if window_type == 'hann':
            window = np.hanning(n)
        elif window_type == 'hamming':
            window = np.hamming(n)
        elif window_type == 'blackman':
            window = np.blackman(n)
        else:
            window = np.ones(n)
        
        windowed_signal = signal * window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed_signal)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        magnitude = np.abs(fft_result)
        
        # Normalize
        if params.get('normalize', True):
            max_mag = np.max(magnitude)
            if max_mag > 0:
                magnitude = magnitude / max_mag
        
        # Convert to dB if requested
        if params.get('log_scale', False):
            magnitude = 20 * np.log10(magnitude + 1e-12)  # Add small value to avoid log(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs, magnitude, 'b-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)' if params.get('log_scale', False) else 'Magnitude')
        ax.set_title(params.get('title', 'FFT Spectrum'))
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, fs/2])
        
        plt.tight_layout()
        plt.show()
