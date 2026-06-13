
import numpy as np
from blocks.base_block import BaseBlock
from scipy import signal


class ChirpBlock(BaseBlock):
    """A swept-sine (chirp) source block.

    Generates a sinusoid whose instantaneous frequency sweeps from ``f0`` at
    t=0 to ``f1`` at ``t1`` seconds, following the selected ``method``. The
    block is a pure function of time (stateless): no state is carried across
    time steps.
    """

    @property
    def block_name(self):
        return "Chirp"

    @property
    def category(self):
        return "Sources"

    @property
    def b_type(self):
        """Source block - generates output without requiring input."""
        return 0

    @property
    def color(self):
        return "darkmagenta"

    @property
    def requires_inputs(self):
        """Source block: no inputs need to be connected."""
        return False

    @property
    def doc(self):
        return (
            "Generates a swept-frequency cosine (chirp) signal.\n\n"
            "The instantaneous frequency sweeps from f0 (at t=0) to f1 "
            "(at t=t1) using the chosen method.\n\n"
            "y(t) = Amplitude * chirp(t, f0, t1, f1, method)\n\n"
            "Parameters:\n"
            "- f0: Start frequency in Hz (frequency at t=0).\n"
            "- f1: End frequency in Hz (frequency at t=t1).\n"
            "- t1: Time at which the frequency reaches f1 (s).\n"
            "- amplitude: Peak amplitude of the signal.\n"
            "- method: Frequency sweep profile "
            "('linear', 'logarithmic', or 'quadratic').\n\n"
            "Usage:\nStandard test signal for frequency response / "
            "system identification."
        )

    @property
    def params(self):
        return {
            "f0": {"type": "float", "default": 0.0,
                   "doc": "Start frequency in Hz (at t=0)."},
            "f1": {"type": "float", "default": 10.0,
                   "doc": "End frequency in Hz (at t=t1)."},
            "t1": {"type": "float", "default": 10.0,
                   "doc": "Time at which the frequency reaches f1 (s)."},
            "amplitude": {"type": "float", "default": 1.0,
                          "doc": "Peak amplitude of the signal."},
            "method": {
                "type": "choice",
                "default": "linear",
                "options": ["linear", "logarithmic", "quadratic"],
                "doc": "Frequency sweep profile.",
            },
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "float"}]

    def draw_icon(self, block_rect):
        """Draw a swept-sine (increasing frequency) icon in 0-1 coords."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(0.1, 0.5)
        path.quadTo(0.20, 0.15, 0.30, 0.5)
        path.quadTo(0.40, 0.85, 0.50, 0.5)
        path.quadTo(0.575, 0.2, 0.65, 0.5)
        path.quadTo(0.725, 0.8, 0.80, 0.5)
        path.quadTo(0.85, 0.3, 0.90, 0.5)
        return path

    def execute(self, time, inputs, params, **kwargs):
        f0 = float(params.get("f0", 0.0))
        f1 = float(params.get("f1", 10.0))
        t1 = float(params.get("t1", 10.0))
        amplitude = float(params.get("amplitude", 1.0))
        method = params.get("method", "linear")

        # phi=-90 deg makes the chirp a sine (value 0 at t=0) rather than the
        # scipy default cosine, matching amplitude*sin(0)=0 at t=0.
        value = signal.chirp(
            t=time, f0=f0, t1=t1, f1=f1, method=method, phi=-90.0
        )
        return {0: np.array(amplitude * value, dtype=float)}
