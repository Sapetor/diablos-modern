
import numpy as np
from blocks.base_block import BaseBlock

class SineBlock(BaseBlock):
    """
    A block that generates a sine wave.
    """

    @property
    def block_name(self):
        return "Sine"

    @property
    def category(self):
        return "Sources"

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return "Generates a sine wave."

    @property
    def params(self):
        return {
            "amplitude": {"type": "float", "default": 1.0, "doc": "The amplitude of the sine wave."},
            "omega": {"type": "float", "default": 1.0, "doc": "The angular frequency of the sine wave."},
            "init_angle": {"type": "float", "default": 0.0, "doc": "The initial angle of the sine wave."}
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        amplitude = float(params['amplitude'])
        omega = float(params['omega'])
        init_angle = float(params['init_angle'])
        return {0: np.array(amplitude * np.sin(omega * time + init_angle), dtype=float)}
