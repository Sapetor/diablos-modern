
import numpy as np
from blocks.base_block import BaseBlock


class ImpulseBlock(BaseBlock):
    """
    A block that generates a discrete impulse (Dirac delta approximation).

    Outputs value/dt for one simulation time step at the specified delay,
    and 0 elsewhere. The integral over all time equals value.
    """

    @property
    def block_name(self):
        return "Impulse"

    @property
    def category(self):
        return "Sources"

    @property
    def b_type(self):
        return 0

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return (
            "Generates a discrete impulse (Dirac delta approximation)."
            "\n\nOutputs Value/dt for one time step at the Delay time, 0 elsewhere."
            "\nThe integral of the output equals Value."
            "\n\nParameters:"
            "\n- Value: The impulse strength (area under the pulse)."
            "\n- Delay: Time (seconds) when the impulse fires."
            "\n\nUsage:"
            "\nUsed to obtain impulse responses of transfer functions and systems."
        )

    @property
    def params(self):
        return {
            "value": {"type": "float", "default": 1.0, "doc": "Impulse strength (area)."},
            "delay": {"type": "float", "default": 0.0, "doc": "Time when the impulse fires."},
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw impulse icon: vertical spike with arrow."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Baseline
        path.moveTo(0.1, 0.8)
        path.lineTo(0.9, 0.8)
        # Spike
        path.moveTo(0.35, 0.8)
        path.lineTo(0.35, 0.15)
        # Arrowhead
        path.moveTo(0.25, 0.3)
        path.lineTo(0.35, 0.15)
        path.lineTo(0.45, 0.3)
        return path

    def execute(self, time, inputs, params, **kwargs):
        dt = kwargs.get('dtime', params.get('dtime', 0.01))
        delay = float(params.get('delay', 0.0))
        value = float(params.get('value', 1.0))

        if params.get('_init_start_', True):
            params['_impulse_fired'] = False
            params['_init_start_'] = False

        if not params.get('_impulse_fired', False) and time >= delay:
            params['_impulse_fired'] = True
            return {0: np.atleast_1d(np.array(value / dt, dtype=float)), 'E': False}

        return {0: np.atleast_1d(np.array(0.0, dtype=float)), 'E': False}
