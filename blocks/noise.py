
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
        return "Generates Gaussian noise."

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

    def execute(self, time, inputs, params):
        sigma = float(params['sigma'])
        mu = float(params['mu'])
        return {0: np.array(sigma ** 2 * np.random.randn() + mu, dtype=float)}
