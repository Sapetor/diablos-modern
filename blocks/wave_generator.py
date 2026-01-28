
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
                "default": "sine",
                "type": "choice",
                "options": ["sine", "square", "triangle", "sawtooth"]
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

    def execute(self, time, inputs, params):
        wv = params.get("waveform", "sine")
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
        
        if wv == "sine":
            val = np.sin(arg)
        elif wv == "square":
            val = signal.square(arg)
        elif wv == "triangle":
            val = signal.sawtooth(arg, width=0.5)
        elif wv == "sawtooth":
            val = signal.sawtooth(arg, width=1.0)
            
        return {"out": bias + amp * val}

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
