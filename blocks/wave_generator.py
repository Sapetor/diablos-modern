
import numpy as np
from blocks.base_block import BaseBlock
from scipy import signal

class WaveGeneratorBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "WaveGenerator"

    @property
    def category(self):
        return "Sources"

    @property
    def color(self):
        return "darkcyan"

    @property
    def params(self):
        return {
            "waveform": {
                "default": "Sine",
                "type": "choice",
                "options": ["Sine", "Square", "Triangle", "Sawtooth"]
            },
            "amplitude": {"default": 1.0, "type": "float"},
            "frequency": {"default": 1.0, "type": "float"},
            "phase": {"default": 0.0, "type": "float"},
            "bias": {"default": 0.0, "type": "float"}
        }

    @property
    def doc(self):
        return """Generates various waveforms.

Waveforms:
- Sine: Standard sinusoid
- Square: Square wave between -Amp and +Amp
- Triangle: Linear triangle wave
- Sawtooth: Linear sawtooth wave

Output = Bias + Amplitude * Waveform(Frequency * t + Phase)"""

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "float"}]

    def execute(self, time, inputs, params, **kwargs):
        wv = params.get("waveform", "Sine")
        amp = params.get("amplitude", 1.0)
        freq = params.get("frequency", 1.0)
        phase = params.get("phase", 0.0)
        bias = params.get("bias", 0.0)
        
        # t is current simulation time
        t = time
        
        # Argument for periodic functions
        # 2*pi*f*t + phi
        arg = 2 * np.pi * freq * t + phase
        
        val = 0.0
        
        if wv == "Sine":
            val = np.sin(arg)
        elif wv == "Square":
            val = signal.square(arg)
        elif wv == "Triangle":
            val = signal.sawtooth(arg, width=0.5)
        elif wv == "Sawtooth":
            val = signal.sawtooth(arg, width=1.0)
            
        return {0: bias + amp * val}

    def draw_icon(self, block_rect):
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        
        # Draw a composite wave icon (sine + square hint)
        # Sine part
        path.moveTo(0.1, 0.5)
        path.cubicTo(0.2, 0.2, 0.3, 0.2, 0.4, 0.5)
        
        # Square part
        path.lineTo(0.4, 0.2)
        path.lineTo(0.6, 0.2)
        path.lineTo(0.6, 0.8)
        path.lineTo(0.8, 0.8)
        path.lineTo(0.8, 0.5)
        
        return path
