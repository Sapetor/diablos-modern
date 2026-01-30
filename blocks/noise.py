
import numpy as np
from blocks.base_block import BaseBlock

class NoiseBlock(BaseBlock):
    """
    A block that generates Gaussian noise.
    """

    @property
    def block_name(self):
        return "Noise"

    @property
    def category(self):
        return "Sources"

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return (
            "White Noise Generator."
            "\n\nGenerates random numbers with a Normal (Gaussian) distribution."
            "\n\nParameters:"
            "\n- Mean: Average value (center)."
            "\n- Std Dev: Standard Deviation (spread)."
            "\n- Seed: Random seed for reproducibility (0 = random)."
            "\n\nUsage:"
            "\nSimulate sensor noise or process disturbances."
        )

    @property
    def params(self):
        return {
            "mu": {"type": "float", "default": 0.0, "doc": "The mean of the noise."},
            "sigma": {"type": "float", "default": 1.0, "doc": "The standard deviation of the noise."}
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw noise/random signal icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(0.1, 0.5)
        path.lineTo(0.2, 0.3)
        path.lineTo(0.3, 0.7)
        path.lineTo(0.4, 0.4)
        path.lineTo(0.5, 0.6)
        path.lineTo(0.6, 0.2)
        path.lineTo(0.7, 0.8)
        path.lineTo(0.8, 0.5)
        path.lineTo(0.9, 0.6)
        return path

    def execute(self, time, inputs, params, **kwargs):
        sigma = float(params['sigma'])
        mu = float(params['mu'])
        return {0: np.array(sigma ** 2 * np.random.randn() + mu, dtype=float)}

